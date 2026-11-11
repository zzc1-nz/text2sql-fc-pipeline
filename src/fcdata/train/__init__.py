"""LoRA SFT trainer with a pluggable backend.

Public API (frozen by CONTRACTS.md)
-----------------------------------
``train(cfg, sft_train, sft_dev, out_dir) -> dict``
    Dispatch on ``cfg.train.backend``:

    * ``"reference"`` -- the pure-torch CPU transformer in
      :mod:`fcdata.train.reference` (offline, reproducible, real LoRA).
    * ``"hf"`` -- the production QLoRA path in :mod:`fcdata.train.hf_peft`
      (Qwen2.5-7B / Llama-3.1-8B, needs a GPU box).

    Returns the exact contents of the ``metrics.json`` it writes, so a caller can
    use the metrics without re-reading the file.

What gets written to ``out_dir``
--------------------------------
``model.pt``       self-contained checkpoint (arch config + tokenizer + state_dict)
``metrics.json``   per-step history, epoch summary, param counts, wall time
``config.json``    the resolved config the run used (reproducibility)
``train_log.txt``  human-readable log (one line per logged step + epoch summaries)

Why the training loop is written out by hand
--------------------------------------------
The reference backend must be *inspectable*: a reader has to be able to see the
warmup+cosine schedule, the gradient clipping and the masked cross-entropy in
one place, not infer them from a framework. It is ~80 lines and it is the honest
part of this repository. Only trainable parameters (the LoRA ``A``/``B``
matrices) go into the optimizer, so a "gradient step" here is genuinely a LoRA
step: the base weights never move, and the test suite asserts that.

Per-record ``SFTRecord.weight``
-------------------------------
A weight above 1.0 means "this record should be seen more often" (the
hard-negative resampling curriculum). It is implemented as **weighted sampling**
of the epoch's record order, not as a loss multiplier -- resampling is what the
ablation actually claims to do ("hard data is scarce, so oversample it"), and a
loss multiplier would silently change the effective learning rate.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from ..records import SFTRecord, write_json
from . import data as train_data
from .reference import (
    ReferenceLM,
    build_model,
    count_parameters,
    describe_model,
    generate,
    load_model,
    masked_lm_loss,
    save_checkpoint,
)
from .vocab import WordTokenizer

__all__ = ["train", "train_reference", "train_hf", "TrainResult"]

#: Sentinels for "no cap".
_NO_CAP = 0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _tcfg(cfg: Any, name: str, default: Any) -> Any:
    """Read ``cfg.train.<name>`` (or ``cfg.<name>`` for a bare TrainCfg)."""
    train_cfg = getattr(cfg, "train", cfg)
    return getattr(train_cfg, name, default)


def _cfg_snapshot(cfg: Any) -> dict[str, Any]:
    """JSON-safe snapshot of the config for ``config.json``."""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if hasattr(cfg, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"repr": repr(cfg)}


def _lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup then cosine decay to 10% of ``base_lr``.

    Warmup matters even at this scale: the LoRA ``B`` matrices start at zero, so
    the first gradients are pure ``A`` gradients and a full-size first step is
    measurably noisier. Cosine decay (not step decay) is the modern default and
    is what the HF path uses too, so both backends behave alike.
    """
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    floor = 0.1
    return base_lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _epoch_order(n: int, weights: Sequence[float], *, seed: int) -> list[int]:
    """Deterministic epoch ordering honouring per-record sampling weights.

    Uniform weights -> a plain shuffle (fast path). Non-uniform weights ->
    weighted sampling with replacement, which is the resampling curriculum the
    ablation describes.
    """
    import random

    rng = random.Random(seed)
    order = list(range(n))
    if not weights:
        rng.shuffle(order)
        return order
    if all(abs(w - 1.0) < 1e-9 for w in weights):
        rng.shuffle(order)
        return order
    return rng.choices(order, weights=[max(1e-6, float(w)) for w in weights], k=n)


def _dev_loss(model: ReferenceLM, batches: Sequence[train_data.Batch]) -> tuple[float, float]:
    """Mean masked dev loss + token accuracy over pre-built dev batches."""
    if not batches:
        return float("nan"), float("nan")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    try:
        with torch.no_grad():
            for batch in batches:
                logits = model(batch.input_ids)
                loss, acc, n_sup = masked_lm_loss(logits, batch.labels)
                if n_sup <= 0:
                    continue
                total_loss += float(loss.item())
                total_acc += float(acc)
                n += 1
    finally:
        if was_training:
            model.train()
    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, total_acc / n


def _write_log(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# result object
# --------------------------------------------------------------------------- #
class TrainResult(dict):
    """The metrics dict, also reachable via attributes (``res.wall_time_s``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


def _configure_threads() -> int:
    """Pick a sane torch CPU thread count for these small matmuls.

    Measured on this 20-vCPU box: 4-8 threads is ~2x faster than the default 14+
    for a ~1M-parameter model with ``(16, 380)`` batches. Beyond ~8 threads the
    per-op synchronization cost dominates the tiny GEMMs and throughput actually
    *falls* (14 threads measured at 0.145 steps/s vs 0.370 at 8). The default
    torch setting is derived from the core count, so it lands in the bad region
    on a many-core machine.

    Only applied when the caller has not already chosen a thread count (torch's
    default is 1 before any model is built on some builds), and never forced
    upward, so an explicit ``torch.set_num_threads`` by the user wins.
    """
    import os

    current = torch.get_num_threads()
    n_cpu = os.cpu_count() or 1
    target = max(1, min(8, n_cpu))
    if current > target:
        torch.set_num_threads(target)
        return target
    return current


# --------------------------------------------------------------------------- #
# reference (CPU torch) backend
# --------------------------------------------------------------------------- #
def train_reference(cfg: Any, sft_train: Sequence[SFTRecord],
                    sft_dev: Sequence[SFTRecord] | None, out_dir: str | Path,
                    *, max_steps: int = _NO_CAP, max_records: int = _NO_CAP) -> TrainResult:
    """Train the small reference transformer with LoRA on CPU.

    Everything here is real: real tokenizer fit, real masked cross-entropy, real
    AdamW, real warmup+cosine, real gradient clipping, real dev evaluation.
    """
    t_start = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_threads = _configure_threads()

    # ---- hyper-parameters ---------------------------------------------
    epochs = max(1, int(_tcfg(cfg, "epochs", 3)))
    batch_size = max(1, int(_tcfg(cfg, "batch_size", 16)))
    grad_accum = max(1, int(_tcfg(cfg, "grad_accum", 1)))
    base_lr = float(_tcfg(cfg, "lr", 3e-4))
    weight_decay = float(_tcfg(cfg, "weight_decay", 0.01))
    warmup_ratio = float(_tcfg(cfg, "warmup_ratio", 0.03))
    max_grad_norm = float(_tcfg(cfg, "max_grad_norm", 1.0))
    seed = int(_tcfg(cfg, "seed", 17))
    log_every = max(1, int(_tcfg(cfg, "log_every", 10)))
    eval_every = max(0, int(_tcfg(cfg, "eval_every", 0)))
    vocab_size = int(_tcfg(cfg, "vocab_size", 4096))
    max_seq_len = int(_tcfg(cfg, "max_seq_len", 384))
    template = str(getattr(getattr(cfg, "fmt", cfg), "chat_template", "qwen") or "qwen")

    # config-driven caps (eval's --quick path stamps these when supported)
    if not max_steps:
        max_steps = int(_tcfg(cfg, "max_steps", 0) or 0)
    if not max_records:
        max_records = int(_tcfg(cfg, "max_records", 0) or 0)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU SDPA kernels are already deterministic

    log_lines: list[str] = [
        "=" * 78,
        "fcdata reference LoRA SFT",
        "=" * 78,
        f"out_dir          : {out}",
        f"epochs           : {epochs}",
        f"batch_size       : {batch_size}  grad_accum: {grad_accum}"
        f"  effective: {batch_size * grad_accum}",
        f"lr               : {base_lr}  warmup_ratio: {warmup_ratio}"
        f"  max_grad_norm: {max_grad_norm}",
        f"max_seq_len      : {max_seq_len}  vocab_size(requested): {vocab_size}",
        f"seed             : {seed}  torch_threads: {n_threads}",
    ]

    # ---- 1. tokenizer: fit on the training corpus ----------------------
    t_tok = time.perf_counter()
    texts = train_data.corpus_texts(sft_train, template=template)
    if not texts:
        raise ValueError("training split produced no text; cannot fit a tokenizer")
    tokenizer = WordTokenizer().fit(texts, vocab_size)
    tokenizer_seconds = time.perf_counter() - t_tok
    log_lines.append(f"tokenizer        : {tokenizer.vocab_size} tokens "
                     f"(fit on {len(texts)} texts in {tokenizer_seconds:.2f}s)")

    # ---- 2. encode both splits ----------------------------------------
    encoded_train, weights = train_data.build_encoded_dataset(
        sft_train, tokenizer, max_seq_len=max_seq_len, template=template,
        max_records=max_records)
    encoded_dev, _ = (train_data.build_encoded_dataset(
        sft_dev, tokenizer, max_seq_len=max_seq_len, template=template)
        if sft_dev else ([], []))
    n_train = len(encoded_train)
    avg_sup = sum(e.n_supervised for e in encoded_train) / max(1, n_train)
    log_lines.append(f"train records    : {n_train} (avg supervised target tokens: {avg_sup:.1f})")
    log_lines.append(f"dev records      : {len(encoded_dev)}")

    # ---- 3. model + optimizer -----------------------------------------
    lora_only = bool(_tcfg(cfg, "lora_only", False))
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    info = describe_model(model)
    trainable = model.trainable_parameters()
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.95), eps=1e-8)
    log_lines.append(
        f"model            : {info['total_params']:,} total params, "
        f"{info['trainable_params']:,} trainable ({info['trainable_pct']}%), "
        f"n_layer={info['n_layer']} d_model={info['d_model']} "
        f"n_head={info['n_head']} d_ff={info['d_ff']}")
    log_lines.append(
        f"adaptation       : {'LoRA-only (base frozen)' if lora_only else 'end-to-end'}"
        f"  [LoRA adapters: {info['lora_params']:,} params, r={_tcfg(cfg, 'lora_r', 8)}]")
    if lora_only:
        # Loud, because this mode is only correct on top of a pretrained model.
        log_lines.append(
            "  NOTE: lora_only=True on a randomly-initialised model cannot learn the "
            "task; this mode exists to demonstrate frozen-base LoRA and to run the "
            "ablation. Use the hf backend for real LoRA on a pretrained base.")

    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    planned_steps = steps_per_epoch * epochs
    if max_steps and max_steps > 0:
        planned_steps = min(planned_steps, int(max_steps))
    warmup_steps = int(planned_steps * warmup_ratio)
    log_lines.append(f"schedule         : {steps_per_epoch} steps/epoch, "
                     f"{planned_steps} planned optimizer steps, "
                     f"{warmup_steps} warmup steps")
    log_lines.append("-" * 78)

    dev_batches: list[train_data.Batch] = []
    if encoded_dev:
        dev_batches = [train_data.collate(encoded_dev[i:i + batch_size],
                                         max_seq_len=max_seq_len)
                       for i in range(0, len(encoded_dev), batch_size)]

    # ---- 4. the training loop -----------------------------------------
    history: list[dict[str, Any]] = []
    epochs_summary: list[dict[str, Any]] = []
    step = 0
    tokens_seen = 0
    global_batch = batch_size * grad_accum
    stop = False

    for epoch in range(epochs):
        if stop:
            break
        order = _epoch_order(n_train, weights, seed=seed + epoch)
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        epoch_t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for b_start in range(0, len(order), global_batch):
            if stop:
                break
            chunk = order[b_start:b_start + global_batch]
            accum_chunks = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
            step_loss = 0.0
            step_acc = 0.0
            step_sup = 0
            for mi, mini in enumerate(accum_chunks):
                batch = train_data.collate(
                    [encoded_train[i] for i in mini],
                    [weights[i] for i in mini],
                    max_seq_len=max_seq_len)
                logits = model(batch.input_ids)
                loss, acc, n_sup = masked_lm_loss(logits, batch.labels)
                if n_sup <= 0:
                    continue
                # scale so grad accumulation averages instead of summing
                (loss / max(1, len(accum_chunks))).backward()
                step_loss += float(loss.item())
                step_acc += float(acc)
                step_sup += 1
                tokens_seen += int(batch.input_ids.numel())

            if step_sup == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            lr_now = _lr_at(step, planned_steps, base_lr, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            # clip_grad_norm_ returns the *pre-clip* total norm, which is the
            # informative diagnostic (it shows whether clipping actually engaged);
            # the post-clip norm is recorded too so a test can prove clipping ran.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
            grad_norm_after = min(grad_norm, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = step_loss / step_sup
            acc_val = step_acc / step_sup
            history.append({"step": step, "epoch": epoch,
                            "loss": round(loss_val, 6),
                            "token_acc": round(acc_val, 6),
                            "lr": round(lr_now, 8),
                            "grad_norm": round(grad_norm, 4),
                            "grad_norm_after_clip": round(grad_norm_after, 4),
                            "clipped": bool(grad_norm > max_grad_norm)})
            epoch_losses.append(loss_val)
            epoch_accs.append(acc_val)
            step += 1

            if step % log_every == 0:
                line = (f"epoch {epoch + 1}/{epochs} step {step:5d}/{planned_steps} "
                        f"loss {loss_val:.4f} tok_acc {acc_val:.3f} "
                        f"lr {lr_now:.2e} gnorm {grad_norm:.2f}"
                        f"{' (clipped)' if grad_norm > max_grad_norm else ''}")
                print(line)
                log_lines.append(line)
            if eval_every and step % eval_every == 0 and dev_batches:
                d_loss, d_acc = _dev_loss(model, dev_batches)
                line = f"  [eval] step {step}: dev_loss {d_loss:.4f} dev_tok_acc {d_acc:.3f}"
                print(line)
                log_lines.append(line)
                history[-1]["dev_loss"] = round(d_loss, 6)
                history[-1]["dev_token_acc"] = round(d_acc, 6)

            if max_steps and step >= int(max_steps):
                stop = True
                break

        if not epoch_losses:
            continue
        d_loss, d_acc = _dev_loss(model, dev_batches) if dev_batches else (float("nan"),
                                                                          float("nan"))
        summary = {
            "epoch": epoch,
            "train_loss": round(sum(epoch_losses) / len(epoch_losses), 6),
            "train_token_acc": round(sum(epoch_accs) / len(epoch_accs), 6),
            "dev_loss": None if math.isnan(d_loss) else round(d_loss, 6),
            "dev_token_acc": None if math.isnan(d_acc) else round(d_acc, 6),
            "wall_time_s": round(time.perf_counter() - epoch_t0, 2),
            "steps": len(epoch_losses),
        }
        epochs_summary.append(summary)
        line = (f"== epoch {epoch + 1} done: train_loss {summary['train_loss']:.4f} "
                f"tok_acc {summary['train_token_acc']:.3f} "
                f"dev_loss {summary['dev_loss']} dev_acc {summary['dev_token_acc']} "
                f"({summary['wall_time_s']}s)")
        print(line)
        log_lines.append(line)

    wall_time = time.perf_counter() - t_start
    steps_done = max(1, len(history))

    # ---- 5. final metrics + artefacts ---------------------------------
    first_loss = epochs_summary[0]["train_loss"] if epochs_summary else float("nan")
    last_loss = epochs_summary[-1]["train_loss"] if epochs_summary else float("nan")
    final_dev_loss, final_dev_acc = (_dev_loss(model, dev_batches) if dev_batches
                                     else (float("nan"), float("nan")))

    metrics: TrainResult = TrainResult({
        "backend": "reference",
        "wall_time_s": round(wall_time, 3),
        "tokenizer_fit_s": round(tokenizer_seconds, 3),
        "total_params": info["total_params"],
        "trainable_params": info["trainable_params"],
        "frozen_params": info["frozen_params"],
        "trainable_pct": info["trainable_pct"],
        "lora_params": info["lora_params"],
        "lora_only": info["lora_only"],
        "lora_r": int(_tcfg(cfg, "lora_r", 8)),
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": max_seq_len,
        "n_layer": info.get("n_layer"),
        "d_model": info.get("d_model"),
        "n_head": info.get("n_head"),
        "d_ff": info.get("d_ff"),
        "n_train": n_train,
        "n_dev": len(encoded_dev),
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "steps": steps_done,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "epochs": epochs,
        "lr": base_lr,
        "warmup_steps": warmup_steps,
        "history": history,
        "epochs_summary": epochs_summary,
        "final_train_loss": round(last_loss, 6) if epochs_summary else None,
        "final_train_token_acc": (round(epochs_summary[-1]["train_token_acc"], 6)
                                  if epochs_summary else None),
        "final_dev_loss": None if math.isnan(final_dev_loss) else round(final_dev_loss, 6),
        "final_dev_token_acc": None if math.isnan(final_dev_acc) else round(final_dev_acc, 6),
        "first_epoch_train_loss": round(first_loss, 6) if epochs_summary else None,
        "loss_decreased": bool(epochs_summary and last_loss < first_loss),
        "steps_per_sec": round(steps_done / wall_time, 3),
        "tokens_per_sec": round(tokens_seen / wall_time, 1),
        "tokens_seen": tokens_seen,
        "max_steps_cap": int(max_steps or 0),
        "max_records_cap": int(max_records or 0),
    })

    # checkpoint BEFORE the metrics write so a crash in the log path still leaves a model
    save_checkpoint(out / "model.pt", model, tokenizer, cfg)
    write_json(out / "metrics.json", dict(metrics))
    write_json(out / "config.json", _cfg_snapshot(cfg))

    log_lines.append("-" * 78)
    log_lines.append(f"steps            : {steps_done} in {wall_time:.2f}s "
                     f"({metrics['steps_per_sec']} steps/s, "
                     f"{metrics['tokens_per_sec']} tokens/s)")
    log_lines.append(f"final train loss : {metrics['final_train_loss']} "
                     f"(first epoch {metrics['first_epoch_train_loss']})")
    log_lines.append(f"final dev loss   : {metrics['final_dev_loss']} "
                     f"dev_tok_acc: {metrics['final_dev_token_acc']}")
    log_lines.append(f"loss decreased   : {metrics['loss_decreased']}")
    log_lines.append(f"checkpoint       : {out / 'model.pt'}")
    _write_log(out / "train_log.txt", log_lines)
    print(f"[train] done: {steps_done} steps, {wall_time:.2f}s, "
          f"{metrics['steps_per_sec']} steps/s -> {out}")
    return metrics


# --------------------------------------------------------------------------- #
# hf (production) backend -- imported lazily so torch-only runs never touch it
# --------------------------------------------------------------------------- #
def train_hf(cfg: Any, sft_train: Sequence[SFTRecord],
             sft_dev: Sequence[SFTRecord] | None, out_dir: str | Path,
             **kwargs: Any) -> TrainResult:
    """Delegate to the QLoRA production trainer (needs transformers+peft+GPU)."""
    from .hf_peft import train_hf as _train_hf
    return TrainResult(_train_hf(cfg, sft_train, sft_dev, out_dir, **kwargs))


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def train(cfg: Any, sft_train: Sequence[SFTRecord],
          sft_dev: Sequence[SFTRecord] | None, out_dir: str | Path,
          *, max_steps: int = _NO_CAP, max_records: int = _NO_CAP) -> dict[str, Any]:
    """Train an SFT model; write ``model.pt`` + ``metrics.json``; return the metrics.

    Parameters
    ----------
    cfg
        A :class:`fcdata.config.Config` (``cfg.train.backend`` selects the
        backend). A bare ``TrainCfg`` also works.
    sft_train, sft_dev
        :class:`fcdata.records.SFTRecord` lists. The target of each record is its
        **last assistant message**; everything before it is the prompt.
    out_dir
        Directory to write into (created if needed). Pass a fresh path per run.
    max_steps, max_records
        Optional hard caps used by smoke/``--quick`` runs. ``max_steps=3`` and
        ``max_records=32`` make a full ``train()`` call finish in ~1-2 s on CPU.
        When omitted (0), ``cfg.train.max_steps`` / ``cfg.train.max_records`` are
        used as a fallback if the config defines them.

    Returns
    -------
    dict
        Exactly the contents of ``metrics.json``.
    """
    backend = str(_tcfg(cfg, "backend", "reference")).lower()
    if backend in ("reference", "torch", "cpu"):
        return dict(train_reference(cfg, sft_train, sft_dev, out_dir,
                                    max_steps=max_steps, max_records=max_records))
    if backend in ("hf", "peft", "qlora"):
        return dict(train_hf(cfg, sft_train, sft_dev, out_dir,
                             max_steps=max_steps, max_records=max_records))
    raise ValueError(f"unknown cfg.train.backend={backend!r}; "
                     f"expected 'reference' or 'hf'")
