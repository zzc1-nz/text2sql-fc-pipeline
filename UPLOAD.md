# 上传到 GitHub 操作指南

仓库已经在本地准备好（分支 `main`，101 个文件，1 次提交，工作区干净）。
**只差最后一步 push** —— 因为这个沙箱访问不了 github.com，需要你在本机执行。

---

## 方式 A：网页建仓 + 命令行 push（推荐，最稳）

### 第 1 步：在 GitHub 上建一个空仓库

1. 打开 https://github.com/new
2. **Repository name** 填：`text2sql-fc-pipeline`
   （或你喜欢的名字，比如 `function-calling-data-synthesis`）
3. **Description** 填：
   ```
   LLM data-synthesis pipeline for function-calling fine-tuning (Text2SQL/Spider): synthesis → two-gate validation → tool-call formatting → LoRA SFT → real SQLite evaluation
   ```
4. 选 **Public**（简历项目建议公开，招聘方点不进去等于没写）
5. ⚠️ **不要**勾选 "Add a README file"、"Add .gitignore"、"Choose a license"
   —— 仓库里已经有了，勾了会导致 push 冲突
6. 点 **Create repository**

### 第 2 步：在本地仓库目录执行 push

打开 PowerShell，进入项目目录：

```powershell
cd "C:\Users\24657\Documents\deepseek-harness\default-workspace\text2sql-fc-pipeline"
```

先确认一下待提交内容（应该显示 101 files, 1 commit）：

```powershell
git log --oneline
git status
```

然后关联远程仓库并推送（把 `zzc1-nz` 换成你的用户名，仓库名按第 1 步填的改）：

```powershell
git remote add origin https://github.com/zzc1-nz/text2sql-fc-pipeline.git
git push -u origin main
```

第一次 push 会弹出浏览器让你登录 GitHub（Git Credential Manager 已在你机器上配置好），授权后自动完成。

### 第 3 步：验证

打开 https://github.com/zzc1-nz/text2sql-fc-pipeline 确认能访问。

---

## 方式 B：如果 push 时报错

### `remote origin already exists`
说明之前加过远程地址，先删再加：

```powershell
git remote remove origin
git remote add origin https://github.com/zzc1-nz/text2sql-fc-pipeline.git
git push -u origin main
```

### `failed to push some refs` / `rejected`
远程仓库不是空的（可能你建仓时勾了 README）。二选一：

```powershell
# 选项1：强制推送，用本地内容覆盖远程（仓库刚建、没别的内容时最省事）
git push -u origin main --force
```

```powershell
# 选项2：先合并远程再推（远程有你需要的内容时用）
git pull --rebase origin main
git push -u origin main
```

### `Authentication failed`
凭据过期了，清掉重新登录：

```powershell
git credential-manager erase
```
然后重新 `git push`，会再弹一次浏览器授权。

### 网络连不上 github（国内常见）
给 git 配代理（端口按你实际代理软件改，Clash 通常是 7890）：

```powershell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
git push -u origin main
```

取消代理：
```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```

或改用 SSH（需要已上传 SSH key）：
```powershell
git remote set-url origin git@github.com:zzc1-nz/text2sql-fc-pipeline.git
git push -u origin main
```

---

## 第 4 步：完善仓库设置（重要，别跳过）

### 4.1 填 About + Topics

在仓库主页右侧点 **⚙️ About** 的齿轮图标：

**Description**（粘贴）：
```
LLM data-synthesis pipeline for function-calling fine-tuning (Text2SQL/Spider): synthesis → two-gate validation → tool-call formatting → LoRA SFT → real SQLite evaluation
```

**Topics**（逐个添加）：
```
function-calling   text2sql   spider   lora   peft   llm
data-synthesis     fine-tuning   pytorch   evaluation   sqlite   qlora
```

Topics 决定了别人能不能搜到你的仓库，别省。

### 4.2 确认 README 展示正常

GitHub 会自动渲染根目录 `README.md`。检查两点：
- 顶部的**图片**是否能显示（`reports/*.png` 已提交，应该能正常渲染）
- 表格是否对齐（GitHub 支持 GFM 表格）

### 4.3 可选：加个 homepage

如果你之后写了博客/文章讲解这个项目，填到 About 的 Website 里，招聘方很吃这一套。

---

## 提交内容清单（已确认）

| 目录 | 内容 | 是否上传 |
|---|---|---|
| `src/fcdata/` | 44 个源文件，五阶段流水线实现 | ✅ |
| `tests/` | 9 个测试文件，516 个用例 | ✅ |
| `data/fixture/` | 6 个真实 SQLite 库 + 122 条可执行 gold SQL | ✅ |
| `reports/` | 结果表格 + 4 张图 + CSV/JSON | ✅ |
| `docs/validation.md` | 验证方法论 | ✅ |
| `scripts/` | 10 个可复现脚本 | ✅ |
| `scripts/dev/` | 15 个调试/探针脚本（展示调试轨迹） | ✅ |
| `README.md` | 主文档 | ✅ |
| `RESUME.md` | 简历条目（中英双版）+ 面试问答准备 | ✅ |
| `artifacts/` | 运行产物（模型权重、日志） | ❌ 已在 .gitignore |
| `.scratch/` | 临时目录 | ❌ 已在 .gitignore |

**已扫描确认：无 API key、无大文件（最大 1.6MB）、无模型权重。**

---

## 常见追问：要不要把 artifacts 也传上去？

**不要。** 理由：
- 模型权重、训练日志是运行产物，可以 `python -m fcdata.cli all` 重新生成
- 会让仓库膨胀且每次重跑都产生 diff 噪音
- `reports/` 里的结果已经足够展示结论

如果面试官想看实际模型，告诉他一条命令就能复现：
```bash
PYTHONPATH=src python -m fcdata.cli all
```

---

## 上传后自检清单

- [ ] 仓库能公开访问
- [ ] README 首屏能看懂项目做什么（前 20 行有架构图）
- [ ] `reports/cost_vs_partial.png` 能正常显示
- [ ] 点进 `reports/README_TABLES.md` 能看到完整结果表
- [ ] About 的 Description 和 Topics 已填
- [ ] 把链接加到简历和 GitHub 主页 pinned repositories
- [ ] （可选）在 GitHub 主页把该仓库 **Pin** 到最前面

在 GitHub 主页点击 **Customize your pins** → 勾选这个仓库 → Save。
