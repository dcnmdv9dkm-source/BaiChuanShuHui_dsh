# BaiChuanShuHui_dsh

**百川数汇（BaiChuanShuHui）科研数据清洗与统计分析引擎 — 原生 DeepSeek Harness 工具插件**

[English](README.md) | 中文

[![CI](https://img.shields.io/badge/CI-passing-green)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-blue)](#) [![DeepSeek Harness](https://img.shields.io/badge/DeepSeek%20Harness-0.1.0--rc.6-blue)](#)

`BaiChuanShuHui_dsh` 是 **DeepSeek Harness** 的插件。它把百川数汇统计引擎——
一个零依赖 Python 命令行工具（含 **30+ 子命令**，覆盖数据清洗、描述统计、假设检验、
方差分析、回归、IRT、CFA、SEM、功效分析、学科感知可视化等）——以**原生模型工具**的形式暴露出来。
当你让 DeepSeek 清洗数据、做方差分析或绘制学科专属图表时，Harness 会调用对应工具，
由插件在本地运行 Python 内核并以结构化 JSON 返回结果。

## 工作原理

| 层级 | 发生了什么 |
|---|---|
| 模型 | 看到 30+ 个带真实 JSON-Schema 参数的工具（`bcsh_clean`、`bcsh_anova`、`bcsh_regress`、`bcsh_viz` …）。 |
| Harness | 把工具调用路由到本插件的 `execute()`。 |
| 插件 | 启动一个轻量 Python 桥接器（`python/bcsh_run_tool.py`），由它内省内核并执行子命令。 |
| 内核 | `skill_runtime.py` 在本地完成计算，向 stdout 打印统一 JSON 结果。 |

插件**不替换**主模型或任何适配器。工具清单在加载时通过对内核自身 `argparse` 的内省生成，
因此 30 个工具始终与引擎保持一致——无需手工维护 schema。

引擎会自动检测装有 `numpy` / `scipy` / `scikit-learn` 的 Python 并启用更精确算法；
仅用纯 `python3` 时则优雅回退到无依赖的精确实现。

## 安装

使用 DeepSeek Harness 内置的插件管理器：

```sh
npx @deepseek-ai/dsh plugin --profile web add github:dcnmdv9dkm-source/BaiChuanShuHui_dsh
```

重启 Harness，插件会自动注册其工具，无需任何设置卡片。直接对模型说
“清洗 data.csv 中的缺失值与异常值”或“对 score 按 group 做单因素方差分析”，
它就会调用对应工具。

> DeepSeek Harness 仍处于开发者预览阶段。本版本精确对应 `0.1.0-rc.6`。
> 若你的 Harness 处于其他预览版本，请调整 `package.json` 中的 `peerDependencies`。

## 配置

插件需要 Python 解释器与内置内核，二者均自动探测，也可通过 Harness 启动环境（或 `$DSH_HOME`
环境变量）覆盖：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `BCSH_PYTHON` | `python3` | 运行内核的 Python 解释器。指向装有 `numpy`/`scipy`/`scikit-learn` 的虚拟环境可获得最精确算法。 |
| `BCSH_KERNEL_DIR` | `<插件>/python` | 包含 `skill_runtime.py` 的目录。用它指向外部/更新的内核，替代内置内核。 |
| `BCSH_TIMEOUT` | `300` | 内核子进程的单次调用超时（秒）。 |

无需 API Key、无网络出向、无机密需要存储。

## 安全边界

- 内核以**本地子进程**方式运行，只读取模型显式传入的输入文件、写入显式传入的输出路径
  （`--input-file`、`--output` 等）。
- 内核**不进行任何网络调用**。唯一的网络流量是 Harness 自身发出的 DeepSeek API 请求，与本插件无关。
- 工具参数先经内核自身的 `argparse` 校验再执行；未知键会被忽略，而非执行。
- 长时间调用遵守取消语义：被中止的工具调用会通过 Harness 的 `AbortSignal` 杀死内核子进程。
- 仅执行内置的 Python 代码，除引擎本身外不引入任何第三方运行时。

## 开发

```sh
pnpm install
pnpm build      # tsdown → lib/index.js
pnpm typecheck  # tsc --noEmit（完整类型检查需安装 @deepseek-ai 对等类型）
```

Python 部分可独立测试：

```sh
python3 python/bcsh_run_tool.py --list          # 列出 30+ 子命令
python3 python/bcsh_run_tool.py --emit-tools    # 打印 JSON-Schema 工具定义
python3 python/bcsh_run_tool.py clean '{"input":"data.csv"}'
```

本项目基于 [MIT 许可证](LICENSE) 开源。统计内核（`skill_runtime.py`）来自百川数汇技能的
单文件零依赖内核；本插件是构建于其上的、Harness 原生薄桥接层。
