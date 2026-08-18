import { execFileSync, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
//#region src/index.ts
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
const name = "BaiChuanShuHui_dsh";
const inject = ["tools"];
const LIB_DIR = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(LIB_DIR, "..");
const BRIDGE = join(PKG_ROOT, "python", "bcsh_run_tool.py");
/** 解析 Python 解释器：优先环境变量，否则回退 python3。 */
function resolvePython() {
	return process.env.BCSH_PYTHON && process.env.BCSH_PYTHON.trim() !== "" ? process.env.BCSH_PYTHON : "python3";
}
/** 在子进程中运行桥接器并收集 stdout；尊重 AbortSignal 以便取消。 */
function runBridge(args, signal) {
	return new Promise((resolve, reject) => {
		const child = spawn(resolvePython(), [BRIDGE, ...args], {
			env: {
				...process.env,
				PYTHONIOENCODING: "utf-8"
			},
			signal
		});
		let out = "";
		let err = "";
		child.stdout.on("data", (d) => {
			out += d.toString("utf-8");
		});
		child.stderr.on("data", (d) => {
			err += d.toString("utf-8");
		});
		child.on("error", (e) => reject(e));
		child.on("close", (code) => {
			const trimmed = out.trim();
			if (code === 0 && trimmed !== "") resolve(trimmed);
			else reject(new Error((trimmed || err || `bridge exited with code ${code}`).trim()));
		});
	});
}
/**
* 共享输出投影：内核返回统一 JSON 对象，原样交给模型；
* 个别命令输出纯文本时已被桥接器包成 {"status":"ok","raw":...}。
*/
const SHARED_OUTPUT = {
	schema: { type: "object" },
	render: (_args, value) => {
		return [{
			type: "text",
			text: typeof value === "string" ? value : JSON.stringify(value, null, 2)
		}];
	}
};
function apply(ctx) {
	let defs;
	try {
		const raw = execFileSync(resolvePython(), [BRIDGE, "--emit-tools"], {
			encoding: "utf-8",
			maxBuffer: 67108864,
			env: {
				...process.env,
				PYTHONIOENCODING: "utf-8"
			},
			timeout: 12e4
		});
		defs = JSON.parse(raw);
	} catch (err) {
		ctx.logger?.error?.("BaiChuanShuHui_dsh: 无法从内核获取工具清单，插件未注册任何工具。");
		ctx.logger?.error?.(err);
		return;
	}
	for (const def of defs) {
		const sub = def.name.startsWith("bcsh_") ? def.name.slice(5) : def.name;
		ctx.tools.register({
			name: def.name,
			description: def.description,
			parameters: def.parameters,
			output: SHARED_OUTPUT,
			async execute(args, exec) {
				const json = await runBridge([sub, JSON.stringify(args ?? {})], exec?.signal);
				try {
					return JSON.parse(json);
				} catch {
					return {
						status: "ok",
						raw: json
					};
				}
			}
		});
	}
	ctx.logger?.info?.(`BaiChuanShuHui_dsh: 已注册 ${defs.length} 个百川数汇工具`);
}
//#endregion
export { apply, inject, name };
