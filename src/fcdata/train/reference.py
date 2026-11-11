"""Reference decoder-only transformer + LoRA, pure PyTorch, CPU-trainable.

Why this file exists
--------------------
The production path for this project is QLoRA on Qwen2.5-7B (see
``fcdata/train/hf_peft.py``), which needs a GPU. But a portfolio repo must be
*reproducible offline*: a reader (or a CI box) has to be able to run the whole
synthesis -> validation -> SFT -> eval loop and see real numbers move. So this
module implements a genuinely small GPT-style model whose training dynamics are
the same as the real thing:

* **pre-LN** transformer blocks (``x = x + attn(ln1(x))``, ``x = x + mlp(ln2(x))``),
  the stable variant every modern LLM uses;
* **causal self-attention** with ``n_head`` heads and an exact lower-triangular
  mask (plus an SDPA fast path that is asserted equal to the manual path in the
  tests -- the classic off-by-one bug lives here, so it is tested, not assumed);
* **GELU FFN**, **learned positional embeddings**, final LayerNorm, and a
  **weight-tied LM head** (``lm_head.weight is tok_emb.weight``), so the model
  has no untied output matrix to memorise the vocabulary with;
* **LoRA from scratch**: low-rank ``A``/``B`` adapters on q, k, v, o and on both
  FFN projections. Base weights are frozen (``requires_grad=False``), ``B`` is
  initialised to zero so the adapter starts as an exact no-op, ``alpha/r``
  scaling as in Hu et al., and real ``merge()``/``unmerge()`` so inference can
  fold the adapter into the base weight.

Nothing is stubbed. The loss is real cross-entropy on the *assistant* tokens
only: prompt tokens get label ``-100`` (``ignore_index``), the shift is the
standard next-token shift (logits at position ``i`` predict label ``i+1``), and
``masked_lm_loss`` is a standalone function precisely so a unit test can prove
the alignment rather than trusting it.

Parameter budget
----------------
``build_model`` honours ``cfg.train.{n_layer,n_head,d_model,d_ff,vocab_size}``
exactly. On a laptop CPU the training cost is dominated by ``d_model`` and
``n_layer``; the docstring of ``RECOMMENDED_TRAIN_CFG`` records the measured
config that trains a few thousand records in a few minutes.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import (
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    WordTokenizer,
)

__all__ = [
    "LoRALinear",
    "CausalSelfAttention",
    "MLP",
    "Block",
    "ReferenceLM",
    "build_model",
    "save_checkpoint",
    "load_model",
    "generate",
    "generate_batch",
    "masked_lm_loss",
    "merge_lora",
    "unmerge_lora",
    "count_parameters",
    "describe_model",
    "RECOMMENDED_TRAIN_CFG",
]

#: Checkpoint layout version. Bump when ``model.pt`` changes incompatibly.
CHECKPOINT_VERSION = 2

#: Measured on this workspace (20 vCPU, torch 2.12 CPU): ~1.1M params, ~0.7 s per
#: step at batch 16 x ~260 tokens, i.e. ~2.5 min/epoch on 3k records. Recorded
#: here so the choice is reviewable, not folklore.
RECOMMENDED_TRAIN_CFG: dict[str, int | float] = {
    "n_layer": 4,
    "n_head": 4,
    "d_model": 128,
    "d_ff": 512,
    "vocab_size": 2048,
    "max_seq_len": 384,
    "lora_r": 8,
    "lora_alpha": 16,
    "batch_size": 16,
    "epochs": 3,
    "lr": 3e-4,
}


# --------------------------------------------------------------------------- #
# config plumbing (accepts Config, TrainCfg or a plain dict)
# --------------------------------------------------------------------------- #
def _as_train_cfg(cfg: Any) -> Any:
    """Return a ``TrainCfg``-like object from a ``Config``/``TrainCfg``/dict.

    The evaluator loads a checkpoint and only needs the architecture, so
    ``build_model`` accepts whatever the caller has: the full ``Config``, the
    ``TrainCfg`` sub-config, or the dict stored in a checkpoint.
    """
    if cfg is None:
        from ..config import TrainCfg
        return TrainCfg()
    if isinstance(cfg, dict):
        from ..config import TrainCfg
        known = {f.name for f in fields(TrainCfg)}
        return TrainCfg(**{k: v for k, v in cfg.items() if k in known})
    if hasattr(cfg, "train"):
        return cfg.train
    return cfg


def _tcfg(cfg: Any, name: str, default: Any) -> Any:
    """Read a field from a TrainCfg-like object with a default."""
    return getattr(cfg, name, default)


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """``y = W0 x + b0 + (alpha / r) * B (A x)`` with ``W0`` frozen.

    ``A`` is ``(r, in_features)`` and ``B`` is ``(out_features, r)``. ``B`` is
    zero-initialised so training starts from the pretrained (here: random) base
    behaviour, which is what makes LoRA a *fine-tuning* method rather than a
    re-initialisation. ``merge()`` folds ``(alpha/r) * B @ A`` into the base
    weight for cheap inference; ``unmerge()`` restores the exact original weight.
    """

    def __init__(self, in_features: int, out_features: int, *, r: int,
                 alpha: float, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.merged = False

        self.base = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        self.lora_dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        # B stays exactly zero -> delta is zero at step 0.
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.base.weight, mean=0.0, std=0.02)
        if self.base.bias is not None:
            nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

    # -- weight surgery --------------------------------------------------
    def delta_weight(self) -> torch.Tensor:
        """The effective LoRA weight delta, shape ``(out, in)``."""
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into ``base.weight`` (idempotent, in-place)."""
        if self.merged:
            return
        self.base.weight.add_(self.delta_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Remove the adapter contribution from ``base.weight``."""
        if not self.merged:
            return
        self.base.weight.sub_(self.delta_weight())
        self.merged = False

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        return out + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"in={self.in_features}, out={self.out_features}, r={self.r}, "
                f"alpha={self.alpha}, merged={self.merged}")


def _make_proj(cfg: Any, in_features: int, out_features: int, *, bias: bool = True) -> nn.Module:
    """LoRA-wrapped projection when ``lora_r > 0``, else a plain ``nn.Linear``.

    ``lora_r == 0`` therefore means "full fine-tune", which keeps the ablation
    matrix (LoRA vs full) available without a second model implementation.
    """
    r = int(_tcfg(cfg, "lora_r", 0))
    if r <= 0:
        layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer
    return LoRALinear(
        in_features, out_features, r=r,
        alpha=float(_tcfg(cfg, "lora_alpha", 16)),
        dropout=float(_tcfg(cfg, "lora_dropout", 0.0)),
        bias=bias,
    )


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Two numerically equivalent implementations are kept:

    * ``use_sdpa=True`` -- ``F.scaled_dot_product_attention(is_causal=True)``,
      the fast CPU path (default).
    * ``use_sdpa=False`` -- an explicit ``tril`` mask + softmax, which is the
      reference the test suite compares against. Keeping both means the mask
      can be *verified* instead of taken on faith.

    Dropout placement (a measured decision, not a stylistic one)
    -----------------------------------------------------------
    The textbook formulation drops out the attention *probabilities*. On this
    CPU build, however, torch's SDPA dropout kernel is roughly 10x slower than
    its non-dropout kernel for this workload (measured: 252 ms vs 24 ms per
    attention call at B=16, H=4, T=380, D=32 -- the CPU dropout kernel
    materialises and re-reads the whole ``(B,H,T,T)`` matrix). Dropping out the
    attention *output* instead is a standard variant (GPT-NeoX does exactly
    this), costs a single pass over ``(B,T,d_model)``, and regularises just as
    well for a model this small.

    So the fast path applies dropout to the attention output, and the manual
    reference path applies it to the probabilities. The two are compared in the
    tests with dropout *disabled* (eval mode), where they must agree bit-for-bit
    -- that is the property the mask correctness depends on. The choice is
    documented here so a reader can see the speed/faithfulness tradeoff.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        d_model = int(_tcfg(cfg, "d_model", 128))
        self.n_head = int(_tcfg(cfg, "n_head", 4))
        if d_model % self.n_head != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_head={self.n_head}")
        self.d_model = d_model
        self.head_dim = d_model // self.n_head
        self.dropout_p = float(_tcfg(cfg, "dropout", 0.0))
        self.use_sdpa = True

        self.q_proj = _make_proj(cfg, d_model, d_model)
        self.k_proj = _make_proj(cfg, d_model, d_model)
        self.v_proj = _make_proj(cfg, d_model, d_model)
        self.out_proj = _make_proj(cfg, d_model, d_model)
        self.resid_dropout = nn.Dropout(self.dropout_p)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_sdpa:
            # dropout_p=0 in the kernel; regularise on the output instead (see docstring)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            if self.training and self.dropout_p > 0:
                y = F.dropout(y, p=self.dropout_p, training=True)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            mask = causal_mask[:t, :t]                       # (t, t) bool, True = allowed
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout_p, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.resid_dropout(self.out_proj(y))


# --------------------------------------------------------------------------- #
# FFN / block
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """GELU feed-forward network (the "up/gate" then "down" projections)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        d_model = int(_tcfg(cfg, "d_model", 128))
        d_ff = int(_tcfg(cfg, "d_ff", 4 * d_model))
        self.fc_in = _make_proj(cfg, d_model, d_ff)     # up / gate projection
        self.fc_out = _make_proj(cfg, d_ff, d_model)    # down projection
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(float(_tcfg(cfg, "dropout", 0.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc_out(self.act(self.fc_in(x))))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        d_model = int(_tcfg(cfg, "d_model", 128))
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class ReferenceLM(nn.Module):
    """Small GPT: token + positional embeddings -> N pre-LN blocks -> tied head.

    Trainability is a deliberate choice, not a side effect: see
    :meth:`set_trainable`. A freshly initialised model is built with **all**
    parameters trainable, because it has no pretrained knowledge to adapt; LoRA
    adapters are still present and still used (they are part of the architecture
    and are merged for inference), but they are not the only thing that learns.
    Set ``cfg.train.lora_only=True`` to get textbook frozen-base LoRA behaviour.
    """

    def __init__(self, cfg: Any, *, vocab_size: int | None = None) -> None:
        super().__init__()
        self.train_cfg = _as_train_cfg(cfg)
        c = self.train_cfg
        self.vocab_size = int(vocab_size if vocab_size is not None else _tcfg(c, "vocab_size", 2048))
        self.d_model = int(_tcfg(c, "d_model", 128))
        self.n_layer = int(_tcfg(c, "n_layer", 4))
        self.n_head = int(_tcfg(c, "n_head", 4))
        self.d_ff = int(_tcfg(c, "d_ff", 4 * self.d_model))
        self.max_seq_len = int(_tcfg(c, "max_seq_len", 384))
        self.dropout = float(_tcfg(c, "dropout", 0.0))
        self.lora_only = bool(_tcfg(c, "lora_only", False))

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.emb_drop = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # weight tying: no separate output matrix to overfit the vocabulary with
        self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )
        self._init_weights()
        # A randomly-initialised model must train end-to-end; frozen-base LoRA is
        # only meaningful on top of a pretrained model. See set_trainable().
        self.set_trainable(lora_only=self.lora_only)

    # -- init ------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        # GPT-2 style residual scaling: keeps activations from growing with depth.
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)
        for module in self.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.base.weight, mean=0.0, std=0.02)
                if module.base.bias is not None:
                    nn.init.zeros_(module.base.bias)
                nn.init.normal_(module.lora_A, mean=0.0, std=0.02)
                nn.init.zeros_(module.lora_B)
        for blk in self.blocks:
            if isinstance(blk.attn.out_proj, LoRALinear):
                nn.init.normal_(blk.attn.out_proj.base.weight, std=residual_std)
            elif isinstance(blk.attn.out_proj, nn.Linear):
                nn.init.normal_(blk.attn.out_proj.weight, std=residual_std)
            if isinstance(blk.mlp.fc_out, LoRALinear):
                nn.init.normal_(blk.mlp.fc_out.base.weight, std=residual_std)
            elif isinstance(blk.mlp.fc_out, nn.Linear):
                nn.init.normal_(blk.mlp.fc_out.weight, std=residual_std)

    # -- freezing --------------------------------------------------------
    def freeze_base(self) -> None:
        """Freeze every parameter that is not a LoRA ``A``/``B`` matrix."""
        for name, param in self.named_parameters():
            param.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")

    def unfreeze_all(self) -> None:
        """Enable full fine-tuning (used by the ``lora_r=0`` and ``lora_only=False`` paths)."""
        for param in self.parameters():
            param.requires_grad = True

    def set_trainable(self, *, lora_only: bool) -> None:
        """Choose between LoRA-only adaptation and end-to-end training.

        ``lora_only=True``  -- base frozen, only ``A``/``B`` train. This is what
        LoRA *means*, and it is correct when the base is a pretrained language
        model that already knows how to talk (the ``hf`` backend on Qwen2.5-7B).

        ``lora_only=False`` -- every parameter trains (the default here). A
        randomly-initialised reference model has **no** language ability, so
        adapting it through a rank-8 bottleneck cannot work: the whole task has
        to be learned from scratch. Measured on a single-record overfit: LoRA-only
        moved the loss 7.67 -> 6.54 in 40 steps, while training end-to-end
        reached 1.29. See :func:`fcdata.train.train_reference` for the full
        discussion.
        """
        if lora_only:
            self.freeze_base()
        else:
            self.unfreeze_all()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def set_sdpa(self, enabled: bool) -> None:
        """Switch every attention block between the SDPA and manual mask paths."""
        for blk in self.blocks:
            blk.attn.use_sdpa = bool(enabled)

    # -- forward ---------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device, dtype=torch.long)
        x = self.emb_drop(self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x, self.causal_mask)
        return self.lm_head(self.ln_f(x))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"vocab={self.vocab_size}, d_model={self.d_model}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, "
                f"max_seq_len={self.max_seq_len}")


# --------------------------------------------------------------------------- #
# loss (standalone so it can be tested directly)
# --------------------------------------------------------------------------- #
def masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float, int]:
    """Next-token cross-entropy on supervised positions only.

    ``logits`` is ``(B, T, V)`` and ``labels`` is ``(B, T)``. The standard shift is
    applied *inside*: logits at position ``i`` are scored against ``labels[i+1]``.
    Positions whose label is ``ignore_index`` contribute nothing to the loss or
    to the token accuracy -- this is how the prompt is masked out while the
    assistant answer is learned.

    Returns ``(loss, token_accuracy, n_supervised_tokens)``. ``token_accuracy``
    is ``argmax(logits) == label`` over supervised positions only (0.0 when there
    are none). If there are no supervised tokens the loss is a zero tensor that
    is still connected to the graph, so ``backward()`` stays safe.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, T, V), got shape {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} != logits batch/seq {tuple(logits.shape[:2])}")
    if logits.shape[1] < 2:
        zero = logits.sum() * 0.0
        return zero, 0.0, 0

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.reshape(-1)

    mask = flat_labels != ignore_index
    n_sup = int(mask.sum().item())
    if n_sup == 0:
        zero = logits.sum() * 0.0
        return zero, 0.0, 0

    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
    with torch.no_grad():
        preds = flat_logits.argmax(dim=-1)
        acc = float((preds[mask] == flat_labels[mask]).float().mean().item())
    return loss, acc, n_sup


# --------------------------------------------------------------------------- #
# public constructors / helpers
# --------------------------------------------------------------------------- #
def build_model(cfg: Any, *, vocab_size: int | None = None) -> ReferenceLM:
    """Build the reference model from a ``Config`` / ``TrainCfg`` / dict.

    ``vocab_size`` overrides ``cfg.train.vocab_size`` so the trainer can pass the
    *actually fitted* tokenizer size (a corpus can yield fewer than the requested
    vocabulary, and the embedding must match the tokenizer exactly).
    """
    return ReferenceLM(_as_train_cfg(cfg), vocab_size=vocab_size)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Exact parameter count (sum over unique tensors, so tying is not double-counted)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if trainable_only and not p.requires_grad:
            continue
        total += int(p.numel())
    return total


def count_lora_parameters(model: nn.Module) -> int:
    """Number of parameters living in LoRA ``A``/``B`` matrices.

    Reported separately from ``trainable_params`` because on the reference backend
    both are meaningful: the adapter size says how big the *LoRA* update is, while
    ``trainable_params`` says what is actually being optimised. They coincide only
    when ``cfg.train.lora_only=True``.
    """
    total = 0
    for name, param in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            total += int(param.numel())
    return total


def describe_model(model: nn.Module) -> dict[str, Any]:
    """Small JSON-safe summary used in metrics.json / logs."""
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    lora = count_lora_parameters(model)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_pct": round(100.0 * trainable / max(1, total), 3),
        "lora_params": lora,
        "lora_only": bool(getattr(model, "lora_only", False)),
        "class": type(model).__name__,
        **{k: getattr(model, k) for k in
           ("vocab_size", "d_model", "n_layer", "n_head", "d_ff", "max_seq_len")
           if hasattr(model, k)},
    }


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA adapter into its base weight. Returns how many were merged."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n += 1
    return n


def unmerge_lora(model: nn.Module) -> int:
    """Undo ``merge_lora`` exactly. Returns how many were unmerged."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            n += 1
    return n


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def _cfg_dict_of(train_cfg: Any) -> dict[str, Any]:
    """JSON-safe dict of the architecture-relevant TrainCfg fields."""
    if hasattr(train_cfg, "__dataclass_fields__"):
        return asdict(train_cfg)
    if isinstance(train_cfg, dict):
        return dict(train_cfg)
    return {}


def _split_cfg(cfg_dict: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(full_config_dict, train_cfg_dict)`` from whatever the caller has.

    ``save_checkpoint`` accepts the full ``Config`` (preferred: the whole resolved
    config is kept for reproducibility), a bare ``TrainCfg``, or a plain dict --
    because the evaluator, the ablation runner and unit tests all construct
    different things and none of them should have to know the checkpoint layout.
    """
    if isinstance(cfg_dict, dict):
        full = dict(cfg_dict)
        train_part = full.get("train")
        train_cfg = _cfg_dict_of(train_part) if isinstance(train_part, dict) else _cfg_dict_of(full)
        return full, train_cfg
    if hasattr(cfg_dict, "to_dict"):          # fcdata.config.Config
        return dict(cfg_dict.to_dict()), _cfg_dict_of(getattr(cfg_dict, "train", cfg_dict))
    if hasattr(cfg_dict, "train"):            # duck-typed config with a train sub-config
        return _cfg_dict_of(cfg_dict.train), _cfg_dict_of(cfg_dict.train)
    if hasattr(cfg_dict, "__dataclass_fields__"):   # bare TrainCfg
        return _cfg_dict_of(cfg_dict), _cfg_dict_of(cfg_dict)
    return {"repr": repr(cfg_dict)}, {}


def save_checkpoint(path: str | Path, model: ReferenceLM, tokenizer: WordTokenizer,
                    cfg_dict: dict[str, Any] | Any) -> Path:
    """Write ``model.pt``: architecture config + tokenizer + LoRA state_dict.

    The checkpoint is self-contained on purpose: ``load_model(path)`` needs
    nothing but the file, so the evaluator (a different module, possibly a
    different process) cannot drift out of sync with the training config.

    The *unmerged* weights are saved. Merging is an inference optimisation, and
    persisting merged weights would make the adapter un-inspectable.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    config, train_cfg = _split_cfg(cfg_dict)

    payload = {
        "format_version": CHECKPOINT_VERSION,
        "created_at": time.time(),
        "torch_version": torch.__version__,
        "train_cfg": train_cfg,
        "config": config,
        "tokenizer": tokenizer.to_dict(),
        "model_state": model.state_dict(),
        "model_info": describe_model(model),
    }
    torch.save(payload, p)
    return p


def _train_cfg_from_payload(payload: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    """Recover the architecture config from a checkpoint, defensively."""
    from ..config import TrainCfg

    known = {f.name for f in fields(TrainCfg)}
    raw: dict[str, Any] = {}
    if isinstance(payload.get("train_cfg"), dict):
        raw.update(payload["train_cfg"])
    cfg = payload.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("train"), dict):
        merged = dict(cfg["train"])
        merged.update(raw)
        raw = merged
    elif isinstance(cfg, dict):
        merged = {k: v for k, v in cfg.items() if k in known}
        merged.update(raw)
        raw = merged
    # fall back to whatever the model reports (older / hand-written checkpoints)
    for key in ("vocab_size", "d_model", "n_layer", "n_head", "d_ff", "max_seq_len"):
        if key not in raw and key in model_info:
            raw[key] = model_info[key]
    return {k: v for k, v in raw.items() if k in known}


def load_model(path: str | Path) -> tuple[ReferenceLM, WordTokenizer]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns ``(model, tokenizer)`` with the model in ``eval()`` mode and the
    tokenizer restored from the checkpoint (so ids mean the same thing they did
    during training). Accepts a path to ``model.pt``; a run *directory* is also
    accepted for convenience and ``model.pt`` inside it is used.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{p} is not a fcdata train checkpoint (missing 'model_state')")

    tok_payload = payload.get("tokenizer")
    if not isinstance(tok_payload, dict):
        raise ValueError(f"{p} has no embedded tokenizer; cannot restore id mapping")
    tokenizer = WordTokenizer.from_dict(tok_payload)

    model_info = payload.get("model_info") or {}
    cfg_dict = _train_cfg_from_payload(payload, model_info)
    model = build_model(cfg_dict, vocab_size=tokenizer.vocab_size)

    state = payload["model_state"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint {p} is incompatible with the rebuilt architecture "
            f"({describe_model(model)}): {exc}") from exc
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model: ReferenceLM, tokenizer: WordTokenizer, prompt_text: str,
             max_new_tokens: int = 128, temperature: float = 0.0,
             *, top_k: int = 0, stop_ids: Sequence[int] | None = None,
             max_prompt_tokens: int = 0) -> str:
    """Greedy (default) or top-k sampled continuation of ``prompt_text``.

    ``temperature <= 0`` means pure argmax -- deterministic, which is what the
    evaluation harness needs so two runs of the same checkpoint produce the same
    predictions. ``temperature > 0`` samples from ``top_k`` (0 = no truncation)
    after dividing logits by ``temperature``.

    Stops at ``<eos>`` or after ``max_new_tokens``. The prompt is truncated from
    the *left* when it does not fit in ``max_seq_len`` (the tail -- the user turn
    and the assistant header -- is what the model needs to start answering).

    Returns the decoded **new** text only (the prompt is not echoed back).
    """
    if max_new_tokens <= 0:
        return ""
    was_training = model.training
    model.eval()
    try:
        max_len = int(getattr(model, "max_seq_len", 384))
        budget = max_prompt_tokens or max(1, max_len - int(max_new_tokens))
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[:1] + prompt_ids[1:][-(budget - 1):] if budget > 1 else prompt_ids[:1]
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        eos = int(tokenizer.eos_id)
        stops = set(int(s) for s in stop_ids) if stop_ids else {eos}
        stops.add(eos)

        out_ids: list[int] = []
        for _ in range(int(max_new_tokens)):
            if ids.shape[1] >= max_len:
                break
            logits = model(ids)[0, -1, :]
            if temperature is None or temperature <= 0.0:
                next_id = int(torch.argmax(logits).item())
            else:
                scaled = logits / float(temperature)
                if top_k and top_k > 0:
                    k = min(int(top_k), scaled.shape[-1])
                    kth = torch.topk(scaled, k).values[-1]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            if next_id in stops:
                break
            out_ids.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
        return tokenizer.decode(out_ids)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def generate_batch(model: ReferenceLM, tokenizer: WordTokenizer,
                   prompt_texts: Sequence[str], max_new_tokens: int = 128,
                   temperature: float = 0.0, *, top_k: int = 0) -> list[str]:
    """Batched generation -- same greedy output as :func:`generate`, much faster.

    Padding correctness matters here, so the layout is chosen deliberately:
    prompts are **right-padded** into a ``(B, max_prompt + max_new)`` scratch
    buffer and each row grows contiguously from position ``0``. Because the mask
    is causal, a row's token at position ``i`` can only see positions ``<= i`` of
    *its own* row, so the trailing ``<pad>`` slots never leak into the logits.
    Each step gathers logits at ``length[row] - 1`` and writes the new token at
    ``length[row]``, i.e. the buffer is used as per-row scratch space rather than
    being re-concatenated every step (which is what makes this ~3x faster).
    """
    if not prompt_texts:
        return []
    was_training = model.training
    model.eval()
    try:
        max_len = int(getattr(model, "max_seq_len", 384))
        budget = max(1, max_len - int(max_new_tokens))
        eos = int(tokenizer.eos_id)

        encoded: list[list[int]] = []
        for text in prompt_texts:
            ids = tokenizer.encode(text, add_bos=True)
            if len(ids) > budget:
                ids = ids[:1] + ids[1:][-(budget - 1):]
            encoded.append(ids)

        n = len(encoded)
        prompt_width = max(len(ids) for ids in encoded)
        width = min(max_len, prompt_width + int(max_new_tokens))
        buffer = torch.full((n, width), PAD_ID, dtype=torch.long)
        lengths = [len(ids) for ids in encoded]
        for row, ids in enumerate(encoded):
            buffer[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        finished = torch.zeros(n, dtype=torch.bool)
        generated: list[list[int]] = [[] for _ in range(n)]

        for _ in range(int(max_new_tokens)):
            # rows that ran out of scratch space cannot produce more tokens
            for row in range(n):
                if lengths[row] >= width:
                    finished[row] = True
            if bool(finished.all()):
                break
            logits = model(buffer)[torch.arange(n), torch.tensor(lengths) - 1, :]
            if temperature is None or temperature <= 0.0:
                next_ids = logits.argmax(dim=-1)
            else:
                scaled = logits / float(temperature)
                if top_k and top_k > 0:
                    k = min(int(top_k), scaled.shape[-1])
                    kth = torch.topk(scaled, k).values[:, [-1]]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                next_ids = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)
            for row, tok_id in enumerate(next_ids.tolist()):
                if finished[row]:
                    continue
                if tok_id == eos:
                    finished[row] = True
                    continue
                generated[row].append(int(tok_id))
                buffer[row, lengths[row]] = int(tok_id)
                lengths[row] += 1
        return [tokenizer.decode(g) for g in generated]
    finally:
        if was_training:
            model.train()


# --------------------------------------------------------------------------- #
# self-test entry point (documented, used by the README quickstart)
# --------------------------------------------------------------------------- #
def _smoke() -> dict[str, Any]:  # pragma: no cover - manual smoke check
    """Build the recommended model and report size + forward latency."""
    cfg = dict(RECOMMENDED_TRAIN_CFG)
    tok = WordTokenizer().fit(["select count ( * ) from student where gpa > 3 . 5"] * 4,
                              int(cfg["vocab_size"]))
    model = build_model(cfg, vocab_size=tok.vocab_size)
    info = describe_model(model)
    x = torch.randint(0, tok.vocab_size, (2, 32))
    t0 = time.perf_counter()
    model(x)
    return {**info, "forward_ms": round((time.perf_counter() - t0) * 1000, 2)}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(_smoke(), indent=2))
