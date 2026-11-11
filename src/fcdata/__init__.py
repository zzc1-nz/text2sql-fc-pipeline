"""fcdata: function-calling fine-tuning data synthesis + evaluation pipeline.

Text2SQL is used as the concrete task carrier (Spider format).

Package layout
--------------
fcdata.config        typed config loading (yaml + CLI overrides)
fcdata.tokens        token estimation + budget ledger (the "token investment" accounting)
fcdata.schema        Spider schema model, DDL rendering, schema linking helpers
fcdata.sqlutils      SQL normalization / analysis without external parsers
fcdata.dbexec        real SQLite execution + result-set comparison (exec accuracy)
fcdata.spider        dataset loaders (real Spider format, plus offline fixture)
fcdata.providers     LLM providers (async OpenAI-compatible, deterministic mock, disk cache)
fcdata.synth         data synthesis: CoT, paraphrase, error->correction hard negatives
fcdata.validate      multi-stage validation: format + semantic judge
fcdata.format        function-calling schema conversion -> chat SFT records
fcdata.train         LoRA SFT (numpy/torch reference impl + HF/PEFT production impl)
fcdata.eval          inference + exec accuracy / exact match + ablation runner
fcdata.report        markdown tables + matplotlib figures
"""

__version__ = "0.1.0"
