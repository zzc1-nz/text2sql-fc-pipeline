# 上传到 GitHub 操作指南（PAT Classic 方式）

仓库已就绪：分支 `main`，102 个文件，2 次提交，工作区干净。
推送需要在**你自己的终端**执行（沙箱网络不通）。

---

## ⚠️ 第 0 步：安全处理（必读）

之前有一个 PAT 被贴进过聊天记录 —— **视为已泄露，必须先作废**：

1. 打开 https://github.com/settings/tokens
2. 找到那个 token → **Delete**
3. 每次用完 PAT 都建议随手删，用的时候再建

---

## 第 1 步：创建 PAT (Classic)

1. 打开 https://github.com/settings/tokens/new （登录后直达新建页）
2. **Note** 填：`text2sql push` （随便写，方便识别）
3. **Expiration** 选 `7 days`（用完即弃，最安全）
4. **Select scopes** 只勾一个：
   - ☑️ **repo** （Full control of private repositories）
     - 推送/建仓只需要它。`workflow`、`admin` 等都不用勾
5. 拉到底点 **Generate token**
6. 页面会显示 `ghp_` 开头的 token —— **只显示这一次**，立刻复制

> Classic PAT 建公开仓库 + push，`repo` scope 足够。
> （想更小权限可以用 Fine-grained token：只授权这一个仓库 + Contents: Read and write，但 Classic 更省事。）

---

## 第 2 步：本机推送（二选一）

### 方式 A：标准 git 流程（token 由 git 弹窗收，不进命令行历史）✅ 推荐

打开 PowerShell：

```powershell
cd "C:\Users\24657\Documents\deepseek-harness\default-workspace\text2sql-fc-pipeline"
```

如果你在代理环境下（Clash 等），先给这个仓库配代理（只影响本仓库）：

```powershell
git config http.proxy http://127.0.0.1:7897
```

关联远程并推送：

```powershell
git remote add origin https://github.com/zzc1-nz/text2sql-fc-pipeline.git
git push -u origin main
```

弹出凭据窗口时：
- **Username**：`zzc1-nz`
- **Password**：**粘贴 PAT**（不是 GitHub 登录密码！）

> 如果走的是终端提示（没有弹窗），同样：Username = `zzc1-nz`，Password = PAT。
> 这样 token 只存在于 Git Credential Manager 里，**不会进 shell 历史**。

成功后看到 `Branch 'main' set up to track remote branch 'main'`。

### 方式 B：一键脚本（自动建仓 + 配 Topics + 推送）

```powershell
cd "C:\Users\24657\Documents\deepseek-harness\default-workspace\text2sql-fc-pipeline"
python scripts\dev\push_to_github.py
```

提示 `GitHub PAT (hidden input)` 时粘贴 PAT 回车（隐藏输入，不进历史）。
脚本会自动：探测本地代理端口 → 验证身份 → 建公开仓库 → 写 About 描述 → 设置 12 个 Topics → push → 设默认分支。

前提：代理客户端（Clash Verge 等）要开着，脚本会自动试 7897 / 7890 / 10809。

---

## 第 3 步：验证 + 收尾

1. 打开 https://github.com/zzc1-nz/text2sql-fc-pipeline 确认可见
2. 回到 https://github.com/settings/tokens **删除刚才的 PAT**
3. 填 About/Topics（方式 A 才需要；方式 B 已自动完成）：

   仓库页右侧 **⚙ About** → Description 粘贴：
   ```
   LLM data-synthesis pipeline for function-calling fine-tuning (Text2SQL/Spider): synthesis → two-gate validation → tool-call formatting → LoRA SFT → real SQLite evaluation
   ```
   Topics 逐个添加：
   ```
   function-calling  text2sql  spider  lora  peft  llm  data-synthesis  fine-tuning  pytorch  evaluation  sqlite  qlora
   ```
4. GitHub 主页 **Customize your pins** → Pin 这个仓库

---

## 常见报错

| 报错 | 原因 / 处理 |
|---|---|
| `Authentication failed` | Password 处没粘 PAT，或粘成了登录密码 |
| `The remote end hung up` | 代理没开或端口不对 → `git config http.proxy http://127.0.0.1:7897`（改成你的端口） |
| `support for password authentication was removed` | 粘的是密码不是 PAT；GitHub 已不接受密码推 git |
| `remote origin already exists` | `git remote remove origin` 后重加 |
| `rejected (fetch first)` | 远程非空（建仓勾了 README）→ `git push -u origin main --force` |
| `schannel: AcquireCredentialsHandle failed` | Windows TLS 后端问题 → `git config http.sslBackend openssl` |
| `could not read Username ... terminal prompts disabled` | 加了 `GIT_TERMINAL_PROMPT=0`；去掉它或用方式 B |

---

## 提交内容清单（已确认）

| 目录 | 内容 | 是否上传 |
|---|---|---|
| `src/fcdata/` | 44 个源文件，五阶段流水线 | ✅ |
| `tests/` | 9 个文件，516 个用例 | ✅ |
| `data/fixture/` | 6 个真实 SQLite 库 + 122 条 gold SQL | ✅ |
| `reports/` | 结果表 + 4 图 + CSV/JSON | ✅ |
| `docs/validation.md` | 验证方法论 | ✅ |
| `scripts/` | 10 个可复现脚本 | ✅ |
| `scripts/dev/` | 15 个调试探针 + 一键推送脚本 | ✅ |
| `README.md` / `RESUME.md` / `LICENSE` / `requirements.txt` | — | ✅ |
| `artifacts/` `.scratch/` | 运行产物 | ❌ gitignore |

已扫描：无 API key、无大文件、无模型权重。
