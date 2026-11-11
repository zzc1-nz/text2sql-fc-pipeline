"""Typed configuration for the pipeline.

All stages read a single ``Config`` object built from ``configs/*.yaml`` plus
``--set key=value`` style CLI overrides. Keeping one config object means the
token ledger, the synthesis budget and the ablation runner always agree about
which model was used for what.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    root: str = "."
    data_dir: str = "data"
    fixture_dir: str = "data/fixture"
    spider_dir: str = "data/spider"          # real Spider (optional, see README)
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    cache_dir: str = "artifacts/llm_cache"

    def resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return Path(self.root) / path

    # convenience accessors -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "01_raw"

    @property
    def synth_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "02_synth"

    @property
    def validated_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "03_validated"

    @property
    def sft_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "04_sft"

    @property
    def train_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "05_train"

    @property
    def eval_dir(self) -> Path:
        return self.resolve(self.artifacts_dir) / "06_eval"

    @property
    def report_path(self) -> Path:
        return self.resolve(self.reports_dir)


@dataclass
class ProviderCfg:
    """One logical model role. Roles decouple 'cheap generator' from 'quality judge'."""
    name: str = "mock"
    model: str = "mock-cheap-1"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 512
    concurrency: int = 8
    max_retries: int = 3
    timeout_s: float = 60.0
    # price in USD per 1M tokens, used for the cost side of the cost/benefit table
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0


@dataclass
class SynthCfg:
    # how many augmented records to aim for, per source example, per generator
    cot_per_example: int = 1
    paraphrase_per_example: int = 2
    hard_negative_per_example: int = 1
    # sampling / robustness
    max_source_examples: int = 0            # 0 = all
    seed: int = 17
    # hard-negative curriculum: fraction of examples chosen by difficulty score
    hard_negative_use_difficulty: bool = True
    hard_negative_hard_fraction: float = 0.5
    # safety valve: stop spending once the ledger crosses this (0 = unlimited)
    max_total_tokens: int = 0
    max_cost_usd: float = 0.0


@dataclass
class ValidateCfg:
    enable_format: bool = True
    enable_semantic: bool = True
    # only validate a sample of records with the expensive judge (0 = all)
    semantic_sample: int = 0
    semantic_threshold: float = 0.5
    judge_role: str = "judge"
    # drop or keep records that fail the semantic judge
    drop_failed: bool = True
    max_total_tokens: int = 0


@dataclass
class FormatCfg:
    tool_name: str = "execute_sql"
    chat_template: str = "qwen"              # qwen | llama3 | plain
    system_prompt: str = (
        "You are a SQL assistant. Convert the user's natural-language question into a "
        "single SQLite SELECT statement using the provided database schema."
    )
    include_schema_in_prompt: bool = True
    max_schema_tables: int = 12
    # resampling weight applied to hard-negative records; this is the mechanism
    # the "+ hard negative resampling" ablation arm actually uses, so it lives in
    # config rather than being a magic number inside the formatter.
    hard_negative_weight: float = 1.5
    paraphrase_weight: float = 1.0
    cot_weight: float = 1.0


@dataclass
class TrainCfg:
    backend: str = "reference"              # reference | hf
    # ---- reference (pure-torch) backend: small, CPU-trainable, used offline ----
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    vocab_size: int = 4096
    max_seq_len: int = 384
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Reference backend only. A LoRA-only (frozen-base) update is meaningful when
    # the base is a *pretrained* model -- that is the `hf` backend's job with
    # Qwen2.5-7B. The reference model starts from random weights, so it must be
    # trained end-to-end or it cannot learn the task at all (measured: LoRA-only
    # plateaus at loss ~6.1 while end-to-end reaches ~0.01 on a single example).
    lora_only: bool = False
    # torch CPU thread count for training; 0 = auto (clamped to <=8). On a
    # many-core box the default thread count is 3x SLOWER than 4 threads for
    # models this small, so this is worth exposing.
    torch_threads: int = 0
    # ---- shared ----
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_steps: int = 0                      # 0 = derive from epochs; >0 = hard cap
    seed: int = 17
    log_every: int = 10
    eval_every: int = 0                     # 0 = only at end
    # ---- hf backend (production path; needs transformers+peft+GPU) ----
    hf_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    hf_dtype: str = "bfloat16"
    hf_load_in_4bit: bool = True
    hf_gradient_checkpointing: bool = True


@dataclass
class EvalCfg:
    split: str = "dev"
    max_examples: int = 0
    max_new_tokens: int = 160
    temperature: float = 0.0
    sql_timeout_s: float = 5.0
    num_beams: int = 1


@dataclass
class AblationCfg:
    configs: list[str] = field(
        default_factory=lambda: ["raw_only", "plus_cot", "plus_hard_negatives"]
    )
    dev_sample: int = 0


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    # role -> provider config
    providers: dict[str, ProviderCfg] = field(
        default_factory=lambda: {
            "cheap": ProviderCfg(name="mock", model="mock-cheap-1", price_in_per_mtok=0.05, price_out_per_mtok=0.10),
            "judge": ProviderCfg(
                name="mock", model="mock-judge-1", temperature=0.0, max_tokens=256,
                price_in_per_mtok=1.0, price_out_per_mtok=3.0,
            ),
            "trainer": ProviderCfg(name="mock", model="local-model"),
        }
    )
    synth: SynthCfg = field(default_factory=SynthCfg)
    validate: ValidateCfg = field(default_factory=ValidateCfg)
    fmt: FormatCfg = field(default_factory=FormatCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    ev: EvalCfg = field(default_factory=EvalCfg)
    ablation: AblationCfg = field(default_factory=AblationCfg)
    run_name: str = "demo"
    seed: int = 17

    # ------------------------------------------------------------------ #
    def provider(self, role: str) -> ProviderCfg:
        if role not in self.providers:
            raise KeyError(f"unknown provider role {role!r}; have {sorted(self.providers)}")
        return self.providers[role]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# loading / merging
# --------------------------------------------------------------------------- #
def _coerce(raw: Any, target: Any) -> Any:
    """Coerce a raw scalar into the type of the existing field value."""
    if raw is None:
        return None
    if isinstance(target, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    if isinstance(target, int) and not isinstance(target, bool):
        return int(raw)
    if isinstance(target, float):
        return float(raw)
    if isinstance(target, str):
        return str(raw)
    return raw


def _apply_dict(cfg: Any, patch: dict[str, Any], prefix: str = "") -> None:
    """Recursively apply a dict onto a dataclass instance."""
    for key, value in patch.items():
        if not hasattr(cfg, key):
            raise KeyError(f"unknown config key {prefix + key!r}")
        current = getattr(cfg, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value, prefix + key + ".")
        elif isinstance(current, dict) and isinstance(value, dict):
            # merge nested provider dicts field-wise
            for sub_key, sub_val in value.items():
                if sub_key in current and is_dataclass(current[sub_key]):
                    if isinstance(sub_val, dict):
                        _apply_dict(current[sub_key], sub_val, f"{prefix}{key}.{sub_key}.")
                    else:
                        raise TypeError(f"{prefix}{key}.{sub_key} expects a mapping")
                else:
                    current[sub_key] = sub_val
        else:
            setattr(cfg, key, _coerce(value, current))


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``a.b.c=value`` overrides in order."""
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        # allow json-ish literals for lists / dicts
        raw = raw.strip()
        if raw[:1] in "[{":
            value: Any = json.loads(raw)
        elif raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        elif raw.lower() in {"none", "null"}:
            value = None
        else:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
        parts = dotted.split(".")
        patch: dict[str, Any] = {}
        node = patch
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        _apply_dict(cfg, patch)
    return cfg


def _yaml_load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - yaml is optional
        return json.loads(text)
    return yaml.safe_load(text) or {}


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: Iterable[str] | None = None,
                root: str | os.PathLike[str] | None = None) -> Config:
    cfg = Config()
    if root is not None:
        cfg.paths.root = str(root)
    if path is not None:
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = Path(root) / p
        data = _yaml_load(p)
        # `paths.root` in the yaml is relative to the yaml file's project root
        if "paths" in data and "root" not in data["paths"]:
            data["paths"]["root"] = str(Path(path).resolve().parent.parent) if path else cfg.paths.root
        _apply_dict(cfg, data)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def dump_config(cfg: Config, path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfg.to_json(), encoding="utf-8")


_SCRATCH_ROOT = Path(__file__).resolve().parent.parent.parent / ".scratch"


def scratch_dir(prefix: str = "run", cleanup: bool = False) -> Path:
    """Create a writable scratch directory.

    NOTE: ``tempfile.mkdtemp()`` is unusable in this sandbox -- the directory it
    creates cannot be written into (the platform ACL does not propagate to
    freshly created temp subdirectories). Creating a plain subdirectory under a
    known-writable root works, so all tests and builders use this helper.
    """
    import shutil
    import uuid

    _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    d = _SCRATCH_ROOT / f"{prefix}_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    if cleanup:
        shutil.rmtree(d, ignore_errors=True)
    return d


def ensure_dirs(cfg: Config) -> None:
    for d in (cfg.paths.raw_dir, cfg.paths.synth_dir, cfg.paths.validated_dir,
              cfg.paths.sft_dir, cfg.paths.train_dir, cfg.paths.eval_dir,
              cfg.paths.report_path, cfg.paths.resolve(cfg.paths.cache_dir)):
        Path(d).mkdir(parents=True, exist_ok=True)
