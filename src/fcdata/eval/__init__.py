"""Evaluation package: real-execution accuracy, ablations, and the data-scale curve.

Public API (the CLI calls exactly these)
----------------------------------------
``evaluate_model(cfg, run_dir, spider_data, split_examples, out_dir) -> dict``
    Load a trained checkpoint, greedy-decode every dev question, execute the
    generated SQL against the fixture sqlite, write ``predictions.jsonl`` and
    ``metrics.json``. Contract-frozen signature.

``run_ablation(cfg, spider_data, ablation_cfg, out_dir) -> dict``
    Train one model per data-mix config (raw only / +CoT / +hard negatives) and
    record both the synthesis tokens invested and the measured exec accuracy.

``run_scale_curve(cfg, spider_data, fractions, out_dir) -> dict``
    Same, but sweeping the *amount* of synthetic data instead of its kind.

Everything in this package obeys one rule: an accuracy number is only ever the
mean of real ``exec_match`` outcomes produced by executing SQL in sqlite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..records import Example
from .infer import (
    PromptAdapterError,
    build_prompt_fn,
    evaluate_checkpoint,
    extract_pred_sql,
    load_checkpoint,
    predict_examples,
)
from .metrics import (
    FAILURE_TAXONOMY,
    bootstrap_ci,
    classify_failure,
    format_summary_line,
    summarize,
)

__all__ = [
    "FAILURE_TAXONOMY",
    "PromptAdapterError",
    "bootstrap_ci",
    "build_prompt_fn",
    "classify_failure",
    "evaluate_checkpoint",
    "evaluate_model",
    "extract_pred_sql",
    "format_summary_line",
    "load_checkpoint",
    "predict_examples",
    "run_ablation",
    "run_scale_curve",
    "summarize",
]


def evaluate_model(cfg: Any,
                   run_dir: str | Path,
                   spider_data: Any,
                   split_examples: Sequence[Example],
                   out_dir: str | Path) -> dict[str, Any]:
    """Evaluate ``run_dir/model.pt`` on ``split_examples``; write metrics + predictions.

    Parameters
    ----------
    cfg:
        ``fcdata.config.Config``. Uses ``cfg.ev.max_new_tokens``,
        ``cfg.ev.temperature``, ``cfg.ev.sql_timeout_s`` and ``cfg.ev.max_examples``.
    run_dir:
        A training output directory containing ``model.pt``.
    spider_data:
        ``fcdata.spider.SpiderData``; supplies schemas and the sqlite paths.
    split_examples:
        Examples to evaluate (usually ``spider_data.dev``).
    out_dir:
        Where ``predictions.jsonl`` / ``metrics.json`` land.

    Returns
    -------
    dict
        The ``metrics.json`` payload: ``exec_acc``, ``exact_match``,
        ``executable_rate``, ``n``, bootstrap CIs, per-difficulty / per-db /
        per-kind slices and the failure taxonomy.
    """
    examples = _cap_examples(cfg, split_examples)
    return evaluate_checkpoint(cfg, run_dir, spider_data, examples, out_dir)


def _cap_examples(cfg: Any, examples: Sequence[Example]) -> list[Example]:
    """Apply ``cfg.ev.max_examples`` (0 = all), deterministically."""
    ev = getattr(cfg, "ev", None)
    limit = int(getattr(ev, "max_examples", 0) or 0)
    items = list(examples)
    if limit > 0:
        return items[:limit]
    return items


def run_ablation(cfg: Any,
                 spider_data: Any,
                 ablation_cfg: Any,
                 out_dir: str | Path) -> dict[str, Any]:
    """Lazy re-export of :func:`fcdata.eval.ablation.run_ablation`."""
    from .ablation import run_ablation as _run  # noqa: PLC0415 - avoid torch import cost

    return _run(cfg, spider_data, ablation_cfg, out_dir)


def run_scale_curve(cfg: Any,
                    spider_data: Any,
                    fractions: Sequence[float],
                    out_dir: str | Path) -> dict[str, Any]:
    """Lazy re-export of :func:`fcdata.eval.scale.run_scale_curve`."""
    from .scale import run_scale_curve as _run  # noqa: PLC0415 - avoid torch import cost

    return _run(cfg, spider_data, fractions, out_dir)
