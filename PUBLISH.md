# 发布清单 · BaiChuanShuHui_dsh

把本插件推到 GitHub（`dcnmdv9dkm-source/BaiChuanShuHui_dsh`）后，即可用

```sh
npx @deepseek-ai/dsh plugin --profile web add github:dcnmdv9dkm-source/BaiChuanShuHui_dsh
```

安装。下列步骤**已勾项**为本仓库已完成内容，其余需你在本机/GitHub 操作。

## ✅ 已完成（本仓库内）
- [x] 插件代码、`python/` 内核、`lib/` 构建产物齐备
- [x] `npm run build` 通过，`lib/index.js` 已产出并端到端验证（30 工具注册 + 真实调用）
- [x] `git init` + 初始提交 `53da6f0`（分支 `main`，无 `node_modules`/`__pycache__` 泄漏）
- [x] 命名已统一：owner=`dcnmdv9dkm-source`、repo=`BaiChuanShuHui_dsh`，包名 `@dcnmdv9dkm-source/BaiChuanShuHui_dsh`
- [x] `.gitignore` 已忽略 `node_modules/`、`__pycache__/`、`*.log` 等

## ⏳ 发布前需确认 / 操作

### 1. 推送前改成本机 Git 身份（当前为占位）
本仓库 `user.name`/`user.email` 设的是本地占位（`dcnmdv9dkm-source@users.noreply.github.com`），
**只影响本仓库，不改动你的全局 git 配置**。推送前建议换成你真实的 GitHub 邮箱以便正确归属：
```bash
cd dsh-baichuan-shuhui
git config user.email "你的GitHub邮箱或 <你的ID>+dcnmdv9dkm-source@users.noreply.github.com"
git config user.name  "dcnmdv9dkm-source"
```

### 2. 在 GitHub 新建仓库
- 到 https://github.com/new 创建仓库，名称 `BaiChuanShuHui_dsh`，owner `dcnmdv9dkm-source`。
- 建议：**Public**，不要勾选「Initialize with README/LICENSE/.gitignore」（本仓库已有）。

### 3. 关联远程并推送
```bash
cd dsh-baichuan-shuhui
git remote add origin git@github.com:dcnmdv9dkm-source/BaiChuanShuHui_dsh.git
# 若用 HTTPS： git remote add origin https://github.com/dcnmdv9dkm-source/BaiChuanShuHui_dsh.git
git push -u origin main
```

### 4.（可选）打发布标签
```bash
git tag v0.1.0
git push --tags
```

### 5. 在 DeepSeek Harness 安装并验证
```bash
npx @deepseek-ai/dsh plugin --profile web add github:dcnmdv9dkm-source/BaiChuanShuHui_dsh
```
- 重启 Harness。
- 在对话中让模型调用任一 `bcsh_*` 工具（如 `bcsh_stats`、`bcsh_clean`、`bcsh_discipline`）。
- 首次安装后若工具未出现，确认 Harness 版本为 `0.1.0-rc.6`（本插件对等依赖固定该版）。

## ⚙️ 运行期配置（环境变量，可选）
| 变量 | 默认 | 说明 |
|------|------|------|
| `BCSH_PYTHON` | `python3`（PATH 中） | 运行内核的解释器；指向装有 numpy/scipy/sklearn 的 Python 自动启用精确算法 |
| `BCSH_KERNEL_DIR` | `<插件>/python` | 内核目录；一般无需改（除非内核单独部署） |
| `BCSH_TIMEOUT` | `300` | 单次工具调用超时（秒） |

无需 API Key、无网络出向；数据全部在本地内核处理。

## 📦 是否要发到 npm（可选）
`dsh plugin add github:` 走的是 git 安装，无需 npm 发布。若想额外 `npm publish`（供其他工具链引用），
先确认 `package.json` 的 `publishConfig.access: "public"` 与 `files` 包含 `lib`/`python`/`cordis.patch.yml`，
再 `npm publish`。当前仓库未强制要求此步。
