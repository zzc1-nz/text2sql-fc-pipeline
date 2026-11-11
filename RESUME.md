# 简历条目 / Resume Bullets

> 用法：从下面挑 3–4 条放进简历项目栏。**中文版**适合国内投递，**English version** 适合海外/外企。
> 每条都对应仓库里可验证的产物，面试时可以直接打开文件对照。

---

## 中文简历（推荐 4 条版本）

**Function Calling 微调数据合成与评估流水线（Text2SQL / Spider）**
*Python · PyTorch · LoRA/PEFT · OpenAI API · SQLite*

- 独立设计并实现「数据合成 → 多阶段验证 → Function Calling 格式转换 → LoRA SFT → 真实执行评估」五阶段流水线，以 Spider Text2SQL 为载体，产出可直接用于微调的 function-calling 训练集。
- 用廉价模型批量合成 CoT 推理、同义改写、错误→修正三类增强样本，**合成阶段实测消耗 36.7 万 tokens（约 \$0.19）**，线性外推至 Spider 全量规模为 **2830 万–7210 万 tokens（\$15–38）**，达到千万级 token 投入目标；验证阶段占总成本 94%，据此确定「廉价模型做量、高质量模型做校验」的成本结构。
- 设计双闸门验证：格式闸（SQL 语法 + schema 引用合法性 + 真实 SQLite 执行）与语义闸（LLM judge 判断问题-SQL 一致性），并对困难样本增加「必须真的错」的前置校验；完成率漏斗为 363 → 348 条（95.9%）。
- **发现并修正验证阈值的系统性缺陷**：通过引入独立执行真值校准，证明继承的默认阈值 0.5 会静默丢弃 27% 的正确数据（精确率恒为 100%，阈值仅影响假阴性），改以数据驱动的 0.4 取代，并让验证报告自带假阴性率审计。
- 实现 LoRA SFT 训练（自研 CPU 参考实现 + Qwen2.5-7B QLoRA 生产路径）与基于真实 SQLite 执行的评估，覆盖 exec accuracy / EM 及 6 项部分分指标；通过 8 条样本过拟合到逐字节正确的 SQL，端到端验证分词、标签掩码、因果位移、解码全链路正确性。
- 量化噪声地板（同配置重跑差异 ≈ 1 个样本）以界定效应量，如实报告「合成数据在固定算力下提升 schema 对齐（12.5%→20.8%）但未提升严格执行准确率」，并给出容量瓶颈分析与后续验证路径。

## 中文简历（精简 3 条，适合一页简历）

- 构建 Function Calling 微调数据合成 + 评估流水线（Text2SQL/Spider），覆盖合成、双闸门验证、格式转换、LoRA SFT、真实执行评估五个可独立重跑的阶段；516 个单元测试全绿。
- 合成 CoT/改写/纠错三类样本，实测 36.7 万 tokens（\$0.19），外推 Spider 全量为千万级 tokens；通过执行真值校准发现默认验证阈值会误删 27% 正确数据并修正。
- 基于真实 SQLite 执行评估，以部分分指标定位改进方向：合成数据提升 schema 对齐 12.5%→20.8%，但受模型容量限制未转化执行准确率，如实记录为方向性结论。

---

## English version (3–4 bullets)

**Function-Calling Data Synthesis & Evaluation Pipeline (Text2SQL / Spider)**
*Python · PyTorch · LoRA/PEFT · OpenAI API · SQLite*

- Designed and built a five-stage pipeline — **synthesis → multi-stage validation → function-calling format conversion → LoRA SFT → execution-based evaluation** — producing a directly trainable function-calling corpus for Spider Text2SQL.
- Used a cheap model to synthesise CoT rationales, paraphrases, and error→correction hard negatives: **367K tokens measured (~$0.19)**, extrapolating to **28–72M tokens ($15–38)** at full Spider scale; profiling showed the validation stage accounts for 94% of spend, driving a cheap-generate / quality-verify cost design.
- Built a two-gate validator (format: syntax + schema linking + **real SQLite execution**; semantic: LLM judge) with an extra "must actually be wrong" premise check for hard negatives; end-to-end yield 363 → 348 records (95.9%).
- **Found and fixed a systematic threshold defect**: calibrating against independent execution ground truth showed the inherited threshold silently discarded **27% of provably correct data** (precision was 100% at every threshold — it only controlled false negatives). Replaced with a data-driven 0.4 and made the report self-audit its own false-negative rate.
- Implemented LoRA SFT (from-scratch CPU reference transformer + production Qwen2.5-7B QLoRA path) and execution-based scoring (exec accuracy, exact match, plus six partial-credit metrics). Proved the whole stack correct by overfitting 8 records to **byte-identical gold SQL**.
- Quantified the **noise floor** (same-config rerun spread ≈ 1 example) to bound effect sizes, and reported honestly that synthetic data improved schema grounding (12.5% → 20.8%) without improving strict execution accuracy at this model scale — with a capacity-bound analysis and a clear path to validate further.

---

## 面试可能被追问的点（先准备好答案）

1. **「为什么 exec accuracy 是 0？是不是做失败了？」**
   → 不是。exec accuracy 要求整条 SQL 字节级正确，token 准确率不到 ~99% 就必然是 0。我用 8 条样本过拟合到逐字节正确的 SQL，证明整条链路（分词/掩码/因果位移/解码/工具调用语法）是对的；0 是模型容量问题，不是流水线问题。这也是我加部分分指标的原因。

2. **「千万级 token 是编的吗？」**
   → 不是。演示是真实测量的 36.7 万 tokens（ledger 逐次记录，`artifacts/*_ledger.jsonl` 可查），外推是线性的、可审计的，`python -m fcdata.cli extrapolate` 一条命令复现整张表。

3. **「验证阈值 0.5 是谁定的？」**
   → 我最初定的，而且我错了。teammate 用执行真值反驳我，我自己复现确认后改成 0.4。这条我特别愿意讲，因为它体现的是「用独立真值审计自己的过滤器」这个方法本身。

4. **「为什么不再调参把结果做好看？」**
   → 因为那样就是在拟合这 24 条评测样本。效应量 2 个样本、噪声地板 1 个样本，再调就是自欺。我把噪声地板也测出来写进 README，让读者自己判断。

---

## GitHub 仓库简介（About 栏，建议）

**Description:**
```
LLM data-synthesis pipeline for function-calling fine-tuning (Text2SQL/Spider):
synthesis → two-gate validation → tool-call formatting → LoRA SFT → real SQLite evaluation
```

**Topics:**
```
function-calling  text2sql  spider  lora  peft  llm  data-synthesis
fine-tuning  pytorch  evaluation  sqlite  qlora  python
```
