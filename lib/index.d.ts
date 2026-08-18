//#region src/index.d.ts
/**
 * BaiChuanShuHui_dsh — 百川数汇科研统计分析 · DeepSeek Harness 插件
 *
 * 设计
 * ----
 * 这是一个 **工具型插件 (tool plugin)**：在 ``apply()`` 中同步启动一次 Python 桥接器
 * (``python/bcsh_run_tool.py --emit-tools``) 向内核 introspect 出全部子命令的
 * **原始 JSON-Schema 工具定义**，再逐一 ``ctx.tools.register()`` 注册给 DeepSeek Harness。
 * 由于直接复用内核自身的 argparse 元数据，工具清单永远与内核保持一致，无需手工维护。
 *
 * 每次模型调用某工具时，``execute()`` 以子进程方式启动桥接器，桥接器再子进程启动
 * Python 内核 (``skill_runtime.py``)，把内核打印到 stdout 的统一 JSON 作为工具的
 * 规范返回值返回。插件运行时本身**零 npm 依赖**（仅 Node 内置 child_process），
 * Python 解释器与内核路径可通过环境变量覆盖。
 *
 * 这与 DeepSeek Harness 官方说明一致：``ctx.tools.register()`` 直接接受原始
 * JSON-Schema 工具定义（MCP 来源的工具正是如此进入系统）——见
 * docs/cookbook/extension-cookbook.md 的 "A tool plugin" 一节。
 *
 * @module @dcnmdv9dkm-source/BaiChuanShuHui_dsh
 */
declare const name = "BaiChuanShuHui_dsh";
declare const inject: string[];
declare function apply(ctx: any): void;
//#endregion
export { apply, inject, name };