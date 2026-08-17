#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百川数汇 · DeepSeek Harness 插件桥接器
=====================================

这是 TypeScript 插件 (``src/index.ts``) 与 Python 内核 (``skill_runtime.py``)
之间的薄桥接层。它自身仅依赖标准库：

* ``--emit-tools``  输出可供 ``ctx.tools.register()`` 直接消费的**原始 JSON-Schema**
                     工具定义列表（name / description / parameters）。
* ``--list``        输出全部子命令及其一句话说明。
* ``<cmd> <json>``  以子命令 + 结构化参数执行内核，把内核打印到 stdout 的统一 JSON
                     原样返回；若内核输出了非 JSON 文本，则包成 ``{"status":"ok","raw":...}``。

插件每次工具调用会启动一次本桥接器（再由本桥接器子进程启动内核），因此无需常驻服务，
也天然支持并发与超时取消。
"""

import asyncio
import json
import os
import sys

# 让 bcsh_core 从同目录解析内核（skill_runtime.py 与 disciplines.json 同目录），
# 除非显式设置了 BCSH_KERNEL_DIR 指向外部内核。
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("BCSH_KERNEL_DIR", _HERE)

import bcsh_core  # noqa: E402  (必须在设置 BCSH_KERNEL_DIR 之后导入)


def _normalize(raw: str, cmd: str) -> str:
    """内核输出若不是 JSON，则包成统一结构，避免破坏工具 output.schema。"""
    if raw.lstrip().startswith("{"):
        return raw
    return json.dumps(
        {"status": "ok", "task": cmd, "raw": raw}, ensure_ascii=False
    )


def _emit_tools() -> str:
    out = []
    for name, meta in bcsh_core.TOOLS.items():
        out.append(
            {
                "name": name,
                "description": meta["description"],
                "parameters": meta["schema"],  # 已是 {type:'object', properties, required}
            }
        )
    return json.dumps(out, ensure_ascii=False)


def _run(cmd: str, arguments: dict) -> str:
    key = cmd if cmd.startswith("bcsh_") else "bcsh_" + cmd
    if key not in bcsh_core.TOOLS:
        return json.dumps(
            {"status": "error", "task": cmd, "result": {"message": f"未知子命令: {cmd}"}},
            ensure_ascii=False,
        )
    # 仅转发内核已知参数，未知键忽略，避免被 argparse 当作位置参数。
    known = {dest for (dest, _flag, _kind) in bcsh_core.TOOLS[key]["params"]}
    arguments = {k: v for k, v in (arguments or {}).items() if k in known}
    # 内核以裸子命令名调用（bcsh_ 前缀只是工具名）。
    sub = bcsh_core.TOOLS[key]["cmd"]
    raw = asyncio.run(bcsh_core.run_kernel(sub, arguments))
    return _normalize(raw, cmd)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": "usage: bcsh_run_tool.py <cmd> | --emit-tools | --list [json_args]",
                },
                ensure_ascii=False,
            )
        )
        return 2

    arg = sys.argv[1]
    if arg == "--emit-tools":
        print(_emit_tools())
        return 0
    if arg == "--list":
        print(
            json.dumps(
                {name: meta["description"] for name, meta in bcsh_core.TOOLS.items()},
                ensure_ascii=False,
            )
        )
        return 0

    payload = sys.argv[2] if len(sys.argv) > 2 else "{}"
    try:
        arguments = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(
            json.dumps(
                {"status": "error", "task": arg, "result": {"message": f"参数不是合法 JSON: {exc}"}},
                ensure_ascii=False,
            )
        )
        return 2

    out = _run(arg, arguments)
    sys.stdout.write(out if out.endswith("\n") else out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
