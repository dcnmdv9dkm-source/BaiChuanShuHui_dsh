/**
 * dsh-baichuan-shuhui — 百川数汇科研统计分析 · DeepSeek Harness 插件
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
 * @module @baichuan-shuhui/dsh-baichuan-shuhui
 */

import { spawn, execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

// Cordis 插件约定导出 name / inject / apply；ctx 由宿主注入（这里以 any 承载，
// 避免插件构建期依赖 @deepseek-ai/* 类型包，运行时由 Harness 提供）。
export const name = 'dsh-baichuan-shuhui'
export const inject = ['tools']

// lib/index.js -> package root -> python/bcsh_run_tool.py
const LIB_DIR = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = join(LIB_DIR, '..')
const BRIDGE = join(PKG_ROOT, 'python', 'bcsh_run_tool.py')

/** 解析 Python 解释器：优先环境变量，否则回退 python3。 */
function resolvePython(): string {
  return process.env.BCSH_PYTHON && process.env.BCSH_PYTHON.trim() !== ''
    ? process.env.BCSH_PYTHON
    : 'python3'
}

/** 在子进程中运行桥接器并收集 stdout；尊重 AbortSignal 以便取消。 */
function runBridge(args: string[], signal?: AbortSignal): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const child = spawn(resolvePython(), [BRIDGE, ...args], {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      signal,
    })
    let out = ''
    let err = ''
    child.stdout.on('data', (d: Buffer) => {
      out += d.toString('utf-8')
    })
    child.stderr.on('data', (d: Buffer) => {
      err += d.toString('utf-8')
    })
    child.on('error', (e: Error) => reject(e))
    child.on('close', (code: number | null) => {
      const trimmed = out.trim()
      if (code === 0 && trimmed !== '') resolve(trimmed)
      else reject(new Error((trimmed || err || `bridge exited with code ${code}`).trim()))
    })
  })
}

interface RawToolDef {
  name: string
  description: string
  parameters: Record<string, unknown>
}

/**
 * 共享输出投影：内核返回统一 JSON 对象，原样交给模型；
 * 个别命令输出纯文本时已被桥接器包成 {"status":"ok","raw":...}。
 */
const SHARED_OUTPUT = {
  schema: { type: 'object' } as const,
  render: (_args: unknown, value: unknown) => {
    const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
    return [{ type: 'text', text } as const]
  },
}

export function apply(ctx: any): void {
  // 1) 启动时向内核 introspect 工具清单（一次同步调用）。
  let defs: RawToolDef[]
  try {
    const raw = execFileSync(resolvePython(), [BRIDGE, '--emit-tools'], {
      encoding: 'utf-8',
      maxBuffer: 64 * 1024 * 1024,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      timeout: 120_000,
    })
    defs = JSON.parse(raw)
  } catch (err) {
    ctx.logger?.error?.('dsh-baichuan-shuhui: 无法从内核获取工具清单，插件未注册任何工具。')
    ctx.logger?.error?.(err as Error)
    return
  }

  // 2) 逐一注册为模型可调用工具。
  for (const def of defs) {
    const sub = def.name.startsWith('bcsh_') ? def.name.slice(5) : def.name
    ctx.tools.register({
      name: def.name,
      description: def.description,
      parameters: def.parameters, // 原始 JSON-Schema，Harness 原生接受
      output: SHARED_OUTPUT,
      async execute(args: Record<string, unknown>, exec?: { signal?: AbortSignal }) {
        const json = await runBridge([sub, JSON.stringify(args ?? {})], exec?.signal)
        try {
          return JSON.parse(json)
        } catch {
          return { status: 'ok', raw: json }
        }
      },
    })
  }

  ctx.logger?.info?.(`dsh-baichuan-shuhui: 已注册 ${defs.length} 个百川数汇工具`)
}
