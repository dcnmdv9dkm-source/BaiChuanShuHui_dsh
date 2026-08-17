#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaiChuanShuHui (百川数汇) 共享内核接口层
=========================================

被 MCP server 与 Function-Calling server 共同复用：

* **内省内核 argparse**：在导入时动态派生全部子命令的 tool 元数据
  （name / description / JSON Schema）。内核增删参数时，两套协议暴露的工具定义
  自动同步，无需手工维护两份 schema。
* **入参拼装**：把 tool 的结构化入参拼回内核命令行 argv。
* **执行内核**：通过子进程运行 ``python skill_runtime.py <subcommand> ...``，
  返回内核打印到 stdout 的统一 JSON。

这样无论 harness 走 MCP 还是 OpenAI Function Calling，调用的都是同一套工具、
同一份 schema、同一个内核，行为完全一致。

依赖
----
仅标准库；内核子进程使用的解释器由 ``BCSH_PYTHON`` 控制（默认当前解释器），
指向装有 numpy/scipy/scikit-learn 的 Python 即自动启用更精确算法并优雅回退。
"""

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys

# --------------------------------------------------------------------------- #
# 路径与运行环境
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
# 插件打包时内核与 disciplines.json 同目录；也可由 BCSH_KERNEL_DIR 指向外部内核目录。
KERNEL_DIR = os.environ.get("BCSH_KERNEL_DIR", HERE)
KERNEL = os.path.normpath(os.path.join(KERNEL_DIR, "skill_runtime.py"))
PYTHON = os.environ.get("BCSH_PYTHON", sys.executable)
TIMEOUT = float(os.environ.get("BCSH_TIMEOUT", "300"))


# --------------------------------------------------------------------------- #
# 1) 内省内核 argparse，生成 tool 元数据
# --------------------------------------------------------------------------- #
def _load_kernel_module():
    """以独立模块名导入内核，避免污染当前命名空间（导入不会触发主逻辑）。"""
    spec = importlib.util.spec_from_file_location("bcsh_kernel", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_KMOD = _load_kernel_module()
_PARSER = _KMOD.build_parser()
_SUB = next(a for a in _PARSER._actions if isinstance(a, argparse._SubParsersAction))
_COMMANDS = _SUB.choices  # 子命令名 -> 子解析器
# 子命令的帮助文本存在父解析器的 _choices_actions 上（subparser 自身无 .help）
_CMD_HELP = {c.dest: c.help for c in _SUB._choices_actions if getattr(c, "dest", None)}


def _base_json_type(action):
    """argparse action -> JSON Schema 基础类型。"""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "boolean"
    t = getattr(action, "type", None)
    if t is int:
        return "integer"
    if t is float:
        return "number"
    return "string"


def _is_array(action):
    n = getattr(action, "nargs", None)
    return n in ("*", "+", argparse.REMAINDER) or (isinstance(n, int) and n > 1)


def _build_param(action):
    """把单个 argparse action 映射为 (dest, flag, kind, schema)；返回 None 表示跳过。"""
    if isinstance(action, argparse._HelpAction):
        return None
    if not action.option_strings:
        # 位置参数（罕见，但兼容）
        if action.dest == "cmd":  # 子解析器自身的 dest
            return None
        dest = action.dest
        flag = None
        kind = "array" if _is_array(action) else "value"
    else:
        longs = [o for o in action.option_strings if o.startswith("--")]
        flag = longs[0] if longs else action.option_strings[0]
        dest = flag.lstrip("-").replace("-", "_")
        if isinstance(action, argparse._StoreTrueAction):
            kind = "bool_true"
        elif isinstance(action, argparse._StoreFalseAction):
            kind = "bool_false"
        else:
            kind = "array" if _is_array(action) else "value"

    base = _base_json_type(action)
    schema = {}
    if kind in ("bool_true", "bool_false"):
        schema["type"] = "boolean"
        schema["default"] = (kind == "bool_false")  # store_false 默认 True
    elif kind == "array":
        items = {"type": base}
        if action.choices:
            items["enum"] = [str(c) for c in action.choices]
        schema["type"] = "array"
        schema["items"] = items
    else:
        schema["type"] = base
        if action.choices:
            schema["enum"] = [str(c) for c in action.choices]

    if action.default is not None and action.default is not argparse.SUPPRESS:
        if kind not in ("bool_true", "bool_false"):
            schema["default"] = action.default
    if action.help:
        schema["description"] = action.help
    return (dest, flag, kind, schema)


# TOOLS: tool 名 -> 元数据（两套协议共用）
TOOLS = {}
for _cmd, _sp in _COMMANDS.items():
    _props, _required, _params = {}, [], []
    for _action in _sp._actions:
        _p = _build_param(_action)
        if _p is None:
            continue
        _dest, _flag, _kind, _schema = _p
        _props[_dest] = _schema
        if getattr(_action, "required", False) and _kind not in ("bool_true", "bool_false"):
            _required.append(_dest)
        _params.append((_dest, _flag, _kind))
    TOOLS["bcsh_" + _cmd] = {
        "cmd": _cmd,
        "description": (_CMD_HELP.get(_cmd) or _cmd),
        "schema": {"type": "object", "properties": _props, "required": _required},
        "params": _params,
    }


# --------------------------------------------------------------------------- #
# 2) 把 tool 入参拼回命令行 argv
# --------------------------------------------------------------------------- #
def build_argv(cmd, arguments):
    argv = [cmd]
    specs = {dest: (flag, kind) for (dest, flag, kind) in TOOLS["bcsh_" + cmd]["params"]}
    for dest, val in (arguments or {}).items():
        if val is None:
            continue
        flag, kind = specs.get(dest, (None, "value"))
        if kind == "bool_true":
            if val is True:
                argv.append(flag)
        elif kind == "bool_false":
            if val is False:
                argv.append(flag)
        elif kind == "array":
            items = val if isinstance(val, (list, tuple)) else [val]
            if flag is None:
                argv.extend(str(v) for v in items)
            else:
                argv.append(flag)
                argv.extend(str(v) for v in items)
        else:  # value
            if flag is None:
                argv.append(str(val))
            else:
                argv.append(flag)
                argv.append(str(val))
    return argv


# --------------------------------------------------------------------------- #
# 3) 执行内核子命令，返回 stdout 的统一 JSON 字符串
# --------------------------------------------------------------------------- #
async def run_kernel(cmd, arguments):
    """执行子命令，返回内核 stdout 的 JSON 字符串（出错时包成统一 error JSON）。"""
    argv = build_argv(cmd, arguments)
    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON, KERNEL, *argv,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        return json.dumps({"status": "error", "task": cmd,
                           "result": {"message": f"timeout after {TIMEOUT}s"}}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "task": cmd,
                           "result": {"message": f"execution failed: {exc}"}}, ensure_ascii=False)

    out = stdout.decode("utf-8", "replace").strip()
    err = stderr.decode("utf-8", "replace").strip()
    if not out:
        out = json.dumps({"status": "error", "task": cmd,
                          "result": {"message": err or "no stdout"}}, ensure_ascii=False)
    elif proc.returncode != 0 and not out.lstrip().startswith("{"):
        out = json.dumps({"status": "error", "task": cmd, "returncode": proc.returncode,
                          "result": {"message": err or out}}, ensure_ascii=False)
    return out


# --------------------------------------------------------------------------- #
# 4) 转换为 OpenAI Function-Calling 的 tools 格式
# --------------------------------------------------------------------------- #
def openai_tools():
    """返回 OpenAI / DeepSeek 兼容的 tools 列表（name + description + JSON Schema）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": meta["schema"],
            },
        }
        for name, meta in TOOLS.items()
    ]


if __name__ == "__main__":
    # 直接运行可快速查看已注册工具数量与清单
    print(f"kernel: {os.path.basename(KERNEL)}")
    print(f"tools ({len(TOOLS)}):")
    for name in sorted(TOOLS):
        print(f"  - {name}: {TOOLS[name]['description'][:60]}")
