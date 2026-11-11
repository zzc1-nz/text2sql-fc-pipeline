"""Production QLoRA SFT trainer for Qwen2.5-7B / Llama-3.1-8B (GPU path).

.. warning::

   **This module is NOT exercised in the CPU-only environment of this repo.**
   It requires ``transformers``, ``peft``, ``datasets``, ``accelerate`` and (for
   4-bit QLoRA) ``bitsandbytes`` plus a CUDA GPU with enough VRAM. Those packages
   are not installed here and there is no CUDA build of torch, so the code below
   is written to be *correct and complete* but is only import-guarded here: any
   attempt to call it on a CPU box raises a clear ``RuntimeError`` instead of a
   confusing ``ImportError`` deep inside a stack trace.

   Everything that *can* be validated without a GPU is validated: the loss-mask
   logic lives in :func:`_build_response_labels` / :func:`find_assistant_span`,
   which are pure functions over token ids and are unit-tested against a
   synthetic tokenizer. That is the part most likely to be silently wrong in a
   real QLoRA script, so it is the part that is tested.

Why this file exists at all
---------------------------
The reference backend (``fcdata/train/reference.py``) proves the pipeline end to
end on CPU, but it is a ~1M-parameter model: it cannot answer Spider questions.
The *actual* deliverable of a data-synthesis pipeline is "does this data improve
a real 7B model", which needs this path. It is kept in-repo so the claim is
honest and reproducible on a GPU box, with the exact hyperparameters used.

Design decisions worth calling out
----------------------------------
**Response-only loss masking.** Training on the whole conversation teaches the
model to generate user turns and the schema too. The correct target is the
assistant span only. This is computed by tokenizing the rendered chat and then
locating the assistant header, *not* by tokenizing prompt and answer separately:
subword tokenizers are not concatenation-invariant, so a separate-tokenization
split can shift by a token. :func:`find_assistant_span` scans the token ids for
the template's assistant header and masks everything up to and including it.

**Packing is deliberately off.** Packing multiple examples into one sequence is
faster but leaks attention across example boundaries unless the attention mask is
surgically blocked per example. For a correctness-first reference implementation
the safe default is one example per sequence.

**QLoRA.** ``hf_load_in_4bit`` loads the base model in NF4 with double
quantization and computes in bf16. Only LoRA adapters train, so the optimizer
state is tiny (~40M params for r=16 over all linear layers) instead of ~56GB.

**Gradient checkpointing + accumulation.** Together they make a 7B model fit on a
24GB card at a useful batch size. ``use_reentrant=False`` is required for
gradient checkpointing to work with the LoRA-wrapped modules.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "HF_DEPS",
    "LoraSpec",
    "find_assistant_span",
    "build_response_labels",
    "hf_deps_available",
    "require_hf_deps",
    "train_hf",
    "save_hf_checkpoint",
]

#: Packages this module needs. Kept as data so the error message can name the
#: exact ``pip install`` line instead of just failing.
HF_DEPS: tuple[str, ...] = ("torch", "transformers", "peft", "datasets", "accelerate")

#: Assistant-header strings per chat template. ``find_assistant_span`` locates the
#: last occurrence of the header and masks up to it, so the answer (and its
#: closing turn token) are the only supervised positions.
ASSISTANT_HEADERS: dict[str, str] = {
    "qwen": "<|im_start|>assistant",
    "llama3": "<|start_header_id|>assistant<|end_header_id|>",
    "chatml": "<|im_start|>assistant",
    "plain": "### Assistant:",
}


# --------------------------------------------------------------------------- #
# dependency guard
# --------------------------------------------------------------------------- #
def hf_deps_available() -> tuple[bool, list[str]]:
    """Return ``(all_available, missing_module_names)`` without importing torch GPU."""
    import importlib.util

    missing = [name for name in HF_DEPS if importlib.util.find_spec(name) is None]
    return (not missing, missing)


def require_hf_deps() -> None:
    """Raise a single actionable ``RuntimeError`` when the GPU stack is absent."""
    ok, missing = hf_deps_available()
    if ok:
        return
    raise RuntimeError(
        "the 'hf' training backend needs the production GPU stack, which is not "
        f"installed here (missing: {', '.join(missing)}).\n"
        "This repository's default environment is CPU-only torch, so the reference "
        "backend is used instead (cfg.train.backend='reference').\n"
        "To run the real QLoRA path on a GPU box:\n"
        "  pip install 'transformers>=4.44' 'peft>=0.12' 'datasets>=2.20' "
        "'accelerate>=0.33' 'bitsandbytes>=0.43'\n"
        "and set cfg.train.backend='hf'."
    )


# --------------------------------------------------------------------------- #
# LoRA specification
# --------------------------------------------------------------------------- #
class LoraSpec:
    """The adapter shape, shared by the real run and by the tests.

    Targets all attention projections *and* all MLP projections: on modern
    decoder models, adapting only q/v loses measurable quality, and the MLP
    matrices are where much of the factual knowledge sits.
    """

    TARGET_MODULES: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def __init__(self, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                 target_modules: Sequence[str] | None = None,
                 bias: str = "none") -> None:
        self.r = int(r)
        self.alpha = int(alpha)
        self.dropout = float(dropout)
        self.target_modules = list(target_modules or self.TARGET_MODULES)
        self.bias = bias

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": "CAUSAL_LM",
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"LoraSpec(r={self.r}, alpha={self.alpha}, "
                f"targets={len(self.target_modules)})")


# --------------------------------------------------------------------------- #
# response-only label masking (the part that is unit-tested without a GPU)
# --------------------------------------------------------------------------- #
def find_assistant_span(token_ids: Sequence[int], assistant_header_ids: Sequence[int],
                        *, search_from: int = 0) -> int:
    """Return the index just past the assistant header in ``token_ids``.

    Scans for the *last* occurrence of ``assistant_header_ids`` (the answer we
    want to supervise is the final assistant turn) and returns the position of the
    first token after it. Returns ``-1`` when the header is not present, which
    callers must treat as "cannot determine the span" rather than silently
    supervising the whole sequence.

    This is a pure function over id lists so it can be tested exhaustively without
    a model or a GPU -- and it is the exact operation that, if off by one, trains
    the model to predict the assistant header instead of the answer.
    """
    header = list(assistant_header_ids)
    ids = list(token_ids)
    if not header:
        return -1
    n, h = len(ids), len(header)
    last = -1
    for i in range(max(0, search_from), n - h + 1):
        if ids[i:i + h] == header:
            last = i
    return -1 if last < 0 else last + h


def build_response_labels(input_ids: Sequence[int], assistant_header_ids: Sequence[int],
                          *, ignore_index: int = -100) -> list[int]:
    """Build ``labels`` where only the assistant response is supervised.

    ``labels[i] == -100`` for the system/user turns and the assistant header;
    ``labels[i] == input_ids[i]`` for the response (including its closing turn
    token). The caller still applies the standard causal shift, so this function
    is about *which* positions are eligible, not about alignment.

    Raises ``ValueError`` when the assistant header cannot be found: silently
    returning an all-masked example would waste a training step and hide a
    template mismatch, which is exactly the kind of bug that makes a run look
    fine while learning nothing.
    """
    ids = list(input_ids)
    span = find_assistant_span(ids, assistant_header_ids)
    if span < 0:
        raise ValueError(
            "assistant header not found in the tokenized chat; the chat template "
            "used for rendering does not match the one used for masking "
            f"(header ids={list(assistant_header_ids)})")
    return [ignore_index] * span + [int(t) for t in ids[span:]]


# --------------------------------------------------------------------------- #
# the real trainer
# --------------------------------------------------------------------------- #
def _dtype_from_name(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = (name or "bfloat16").lower()
    if key not in mapping:
        raise ValueError(f"unknown cfg.train.hf_dtype={name!r}; "
                         f"expected one of {sorted(mapping)}")
    return mapping[key]


def _build_quant_config(cfg: Any) -> Any:
    """4-bit NF4 config for QLoRA, or ``None`` when running in full precision."""
    train_cfg = getattr(cfg, "train", cfg)
    if not getattr(train_cfg, "hf_load_in_4bit", False):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = _dtype_from_name(getattr(train_cfg, "hf_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _render_training_text(record: Any, template: str) -> str:
    """Full chat text for one record (prompt + assistant target)."""
    try:
        from ..format import render_messages
    except ImportError as exc:  # pragma: no cover - format is in-repo
        raise RuntimeError(
            "fcdata.format is required by the hf backend to render chat templates"
        ) from exc
    return render_messages(record.messages, template)


def _template_for_model(model_name: str, fallback: str = "qwen") -> str:
    """Pick the chat template name from the model id."""
    name = (model_name or "").lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama3"
    return fallback


def train_hf(cfg: Any, sft_train: Sequence[Any], sft_dev: Sequence[Any] | None,
             out_dir: str | Path, *, max_steps: int = 0,
             max_records: int = 0) -> dict[str, Any]:
    """QLoRA SFT of a real instruction-tuned model.

    Not runnable in this CPU-only environment (see the module docstring); on a GPU
    box it is a complete, correct trainer: NF4 4-bit base, LoRA on all attention
    and MLP projections, response-only loss, gradient checkpointing, gradient
    accumulation, cosine LR with warmup, bf16 compute, adapter checkpoint.

    Returns the same ``metrics.json`` shape as the reference backend, plus
    HF-specific fields, so downstream code (eval, report) does not branch.
    """
    require_hf_deps()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    train_cfg = getattr(cfg, "train", cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_name = str(getattr(train_cfg, "hf_model_name", "Qwen/Qwen2.5-7B-Instruct"))
    dtype = _dtype_from_name(getattr(train_cfg, "hf_dtype", "bfloat16"))
    load_in_4bit = bool(getattr(train_cfg, "hf_load_in_4bit", True))
    grad_ckpt = bool(getattr(train_cfg, "hf_gradient_checkpointing", True))
    template = _template_for_model(model_name,
                                   fallback=str(getattr(getattr(cfg, "fmt", cfg),
                                                        "chat_template", "qwen")))
    epochs = max(1, int(getattr(train_cfg, "epochs", 3)))
    batch_size = max(1, int(getattr(train_cfg, "batch_size", 4)))
    grad_accum = max(1, int(getattr(train_cfg, "grad_accum", 8)))
    lr = float(getattr(train_cfg, "lr", 2e-4))
    warmup_ratio = float(getattr(train_cfg, "warmup_ratio", 0.03))
    max_seq_len = int(getattr(train_cfg, "max_seq_len", 1024))
    seed = int(getattr(train_cfg, "seed", 17))
    log_every = max(1, int(getattr(train_cfg, "log_every", 10)))
    eval_every = max(0, int(getattr(train_cfg, "eval_every", 0)))
    if not max_steps:
        max_steps = int(getattr(train_cfg, "max_steps", 0) or 0)
    if not max_records:
        max_records = int(getattr(train_cfg, "max_records", 0) or 0)

    log_lines = [
        "=" * 78,
        "fcdata QLoRA SFT (production HF/PEFT backend)",
        "=" * 78,
        f"model            : {model_name}",
        f"chat template    : {template}",
        f"quantisation     : {'NF4 4-bit (QLoRA)' if load_in_4bit else 'none'}",
        f"compute dtype    : {getattr(train_cfg, 'hf_dtype', 'bfloat16')}",
        f"grad checkpoint  : {grad_ckpt}",
        f"epochs           : {epochs}",
        f"batch_size       : {batch_size}  grad_accum: {grad_accum}"
        f"  effective: {batch_size * grad_accum}",
        f"lr               : {lr}  warmup_ratio: {warmup_ratio}",
        f"max_seq_len      : {max_seq_len}",
        "-" * 78,
    ]

    # ---- tokenizer -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # decoder-only LMs often have no pad token; reusing EOS is the standard fix
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assistant_header_ids = tokenizer.encode(
        ASSISTANT_HEADERS.get(template, ASSISTANT_HEADERS["qwen"]), add_special_tokens=False)

    # ---- model ---------------------------------------------------------
    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": False}
    quant_config = _build_quant_config(cfg)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False})

    lora = LoraSpec(r=int(getattr(train_cfg, "lora_r", 16)),
                    alpha=int(getattr(train_cfg, "lora_alpha", 32)),
                    dropout=float(getattr(train_cfg, "lora_dropout", 0.05)))
    model = get_peft_model(model, LoraConfig(**lora.as_kwargs()))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_lines.append(f"LoRA             : {lora!r}")
    log_lines.append(f"params           : {total:,} total, {trainable:,} trainable "
                     f"({100.0 * trainable / max(1, total):.3f}%)")
    model.print_trainable_parameters()

    # ---- data: render, tokenize, mask ----------------------------------
    def encode_split(records: Sequence[Any]) -> Dataset:
        rows: list[dict[str, Any]] = []
        for rec in records:
            text = _render_training_text(rec, template)
            enc = tokenizer(text, truncation=True, max_length=max_seq_len,
                            add_special_tokens=False)
            ids = list(enc["input_ids"])
            try:
                labels = build_response_labels(ids, assistant_header_ids)
            except ValueError as exc:
                raise ValueError(f"record {getattr(rec, 'id', '?')}: {exc}") from exc
            if all(lab == -100 for lab in labels):
                continue
            rows.append({"input_ids": ids, "labels": labels,
                         "attention_mask": [1] * len(ids)})
        if not rows:
            raise ValueError("no usable training rows after response-only masking")
        return Dataset.from_list(rows)

    train_records = list(sft_train)
    if max_records and max_records > 0:
        train_records = train_records[:max_records]
    train_ds = encode_split(train_records)
    dev_ds = encode_split(sft_dev) if sft_dev else None
    log_lines.append(f"train examples   : {len(train_ds)}")
    if dev_ds is not None:
        log_lines.append(f"dev examples     : {len(dev_ds)}")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8,
        label_pad_token_id=-100)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (batch_size * grad_accum)))
    training_args = TrainingArguments(
        output_dir=str(out / "hf_checkpoints"),
        num_train_epochs=float(epochs),
        max_steps=int(max_steps) if max_steps else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=float(getattr(train_cfg, "weight_decay", 0.0)),
        max_grad_norm=float(getattr(train_cfg, "max_grad_norm", 1.0)),
        logging_steps=log_every,
        eval_strategy="steps" if (dev_ds is not None and eval_every) else "no",
        eval_steps=eval_every if eval_every else None,
        save_strategy="no",
        bf16=dtype is torch.bfloat16 and torch.cuda.is_available(),
        fp16=dtype is torch.float16 and torch.cuda.is_available(),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=dev_ds, data_collator=collator,
                      tokenizer=tokenizer)

    t0 = time.perf_counter()
    train_output = trainer.train()
    wall = time.perf_counter() - t0

    # ---- save adapter + metrics ---------------------------------------
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    history: list[dict[str, Any]] = []
    for entry in trainer.state.log_history:
        if "loss" in entry and "step" in entry:
            history.append({"step": int(entry["step"]),
                            "epoch": float(entry.get("epoch", 0.0)),
                            "loss": float(entry["loss"]),
                            "lr": float(entry.get("learning_rate", lr))})
        elif "eval_loss" in entry and "step" in entry:
            if history:
                history[-1]["dev_loss"] = float(entry["eval_loss"])

    metrics: dict[str, Any] = {
        "backend": "hf",
        "wall_time_s": round(wall, 3),
        "model_name": model_name,
        "chat_template": template,
        "load_in_4bit": load_in_4bit,
        "hf_dtype": str(getattr(train_cfg, "hf_dtype", "bfloat16")),
        "lora_r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_target_modules": lora.target_modules,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_pct": round(100.0 * trainable / max(1, total), 4),
        "vocab_size": int(len(tokenizer)),
        "max_seq_len": max_seq_len,
        "n_train": len(train_ds),
        "n_dev": len(dev_ds) if dev_ds is not None else 0,
        "steps_per_epoch": steps_per_epoch,
        "steps": int(trainer.state.global_step),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "epochs": epochs,
        "lr": lr,
        "history": history,
        "epochs_summary": [],
        "final_train_loss": float(train_output.training_loss),
        "final_train_token_acc": None,      # Trainer does not track token accuracy
        "final_dev_loss": (float(trainer.state.log_history[-1].get("eval_loss"))
                           if dev_ds is not None and trainer.state.log_history
                           and "eval_loss" in trainer.state.log_history[-1] else None),
        "final_dev_token_acc": None,
        "steps_per_sec": round(int(trainer.state.global_step) / wall, 3) if wall else 0.0,
        "loss_decreased": None,
        "executed_in_this_environment": False,
        "note": ("Production QLoRA path; NOT executed in the CPU-only repo "
                 "environment (requires transformers+peft+bitsandbytes+GPU)."),
    }

    from ..records import write_json

    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json",
               cfg.to_dict() if hasattr(cfg, "to_dict") else {"train": metrics})
    log_lines.append(f"steps            : {metrics['steps']} in {wall:.1f}s "
                     f"({metrics['steps_per_sec']} steps/s)")
    log_lines.append(f"final train loss : {metrics['final_train_loss']}")
    log_lines.append(f"adapter          : {out / 'adapter'}")
    (out / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[train:hf] done: {metrics['steps']} steps in {wall:.1f}s -> {out}")
    return metrics


def save_hf_checkpoint(out_dir: str | Path, model: Any, tokenizer: Any,
                       cfg_dict: dict[str, Any]) -> Path:
    """Save a PEFT adapter + tokenizer + config (GPU path only).

    Kept separate from :func:`train_hf` so a caller that already holds a trained
    model can persist it without re-running training.
    """
    require_hf_deps()
    out = Path(out_dir)
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (out / "config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str), encoding="utf-8")
    return adapter_dir
