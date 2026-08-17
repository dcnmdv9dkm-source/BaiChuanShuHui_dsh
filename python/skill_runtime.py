

from __future__ import annotations
import argparse
import csv
import glob
import html
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import hashlib
from collections import OrderedDict, Counter

HAS_PANDAS = HAS_NUMPY = HAS_SCIPY = HAS_SKLEARN = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    pass
try:
    from scipy import stats as spstats
    HAS_SCIPY = True
except ImportError:
    pass

HAS_SR = False
try:
    from scipy.stats import studentized_range
    HAS_SR = True
except ImportError:
    pass
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    pass

def _sanitize(o):
    # numpy 标量/数组 → Python 原生类型：emit 走 json.dumps(default=str)，
    # 若 numpy 类型直接漏出会被序列化成字符串（"12.34"），破坏数值消费方；此处先行规整。
    mod = getattr(type(o), "__module__", "")
    if mod and mod.split(".")[0] == "numpy":
        try:
            if hasattr(o, "tolist"):
                return _sanitize(o.tolist())
            return _sanitize(o.item())
        except (TypeError, ValueError):
            return str(o)
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return None
        return o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [_sanitize(v) for v in o]
    return o


def _flatten_for_table(d, prefix=""):
    """将嵌套 dict 展平为 {key: value} 行，用于 CSV/MD/HTML 输出。"""
    rows = {}
    for k, v in d.items():
        key = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            rows.update(_flatten_for_table(v, key))
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                rows[key] = "[%d rows]" % len(v)
            else:
                rows[key] = "; ".join(str(x) for x in v[:10])
        else:
            rows[key] = v
    return rows


def _extract_tables(obj):
    """从 emit 对象中提取可表格化的数据，返回 [(title, [col_names], [row_dicts])]。"""
    tables = []
    result = obj.get("result", obj)
    if not isinstance(result, dict):
        return tables
    for key, val in result.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            cols = list(val[0].keys())
            tables.append((key, cols, val))
    if not tables:
        flat = _flatten_for_table(result)
        if flat:
            tables.append(("summary", list(flat.keys()), [flat]))
    return tables


def _emit_csv(obj):
    tables = _extract_tables(obj)
    if not tables:
        sys.stdout.write(json.dumps(_sanitize(obj), ensure_ascii=False, default=str) + "\n")
        return
    for title, cols, rows in tables:
        sys.stdout.write("# %s\n" % title)
        import csv as _csv_mod
        w = _csv_mod.writer(sys.stdout)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
        sys.stdout.write("\n")


def _emit_md(obj):
    tables = _extract_tables(obj)
    if not tables:
        sys.stdout.write("```json\n" + json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, default=str) + "\n```\n")
        return
    for title, cols, rows in tables:
        sys.stdout.write("### %s\n\n" % title)
        sys.stdout.write("| " + " | ".join(cols) + " |\n")
        sys.stdout.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            sys.stdout.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
        sys.stdout.write("\n")


def _emit_html(obj):
    tables = _extract_tables(obj)
    if not tables:
        sys.stdout.write("<pre>" + html.escape(json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, default=str)) + "</pre>\n")
        return
    for title, cols, rows in tables:
        sys.stdout.write("<h3>%s</h3>\n<table>\n<thead><tr>" % html.escape(title))
        for c in cols:
            sys.stdout.write("<th>%s</th>" % html.escape(str(c)))
        sys.stdout.write("</tr></thead>\n<tbody>\n")
        for r in rows:
            sys.stdout.write("<tr>")
            for c in cols:
                sys.stdout.write("<td>%s</td>" % html.escape(str(r.get(c, ""))))
            sys.stdout.write("</tr>\n")
        sys.stdout.write("</tbody></table>\n")


def emit(obj):
    if isinstance(obj, dict):
        copied = False
        if _VERBOSITY != "quiet" and "provenance" not in obj:
            prov = {"kernel_version": _PROV["kernel_version"]}
            if _PROV.get("param_hash") is not None:
                prov["param_hash"] = _PROV["param_hash"]
            if _PROV.get("data_fingerprint") is not None:
                prov["data_fingerprint"] = _PROV["data_fingerprint"]
            obj = dict(obj)
            copied = True
            obj["provenance"] = prov
        if _VERBOSITY != "quiet":
            warns = _dynamic_warnings()
            if warns and "warnings" not in obj:
                if not copied:
                    obj = dict(obj)
                obj["warnings"] = warns
    if _OUTPUT_FORMAT == "csv":
        _emit_csv(obj)
    elif _OUTPUT_FORMAT == "md":
        _emit_md(obj)
    elif _OUTPUT_FORMAT == "html":
        _emit_html(obj)
    else:
        sys.stdout.write(json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, default=str) + "\n")


def die(msg):
    emit({"status": "error", "message": msg})
    sys.exit(1)


def _r4(x):
    if x is None or x != x or x in (float("inf"), float("-inf")):
        return None
    return round(x, 4)

_CFG = {
    "alpha": 0.05,
    "shapiro_max_n": 5000,
    "min_normality_n": 3,
    "skew_rule_cut": 1.0,
    "sentinels": [-99999, -9999, -999, -99, 999, 9999, 99999, -1],
    "mad_scale": 0.6745,
    "if_contamination": {"L1": 0.10, "L2": 0.05, "L3": 0.01},
}


_MISSING_TOKENS = frozenset(("nan", "none", "null", "na", "n/a"))

# ---- 溯源印章（护城河信任硬标签）----
KERNEL_VERSION = "1.7.0"
_PROV = {"kernel_version": KERNEL_VERSION, "param_hash": None, "data_fingerprint": None}

# 运行期告警通道（审计 E1/E2/E4 可见性）：emit 时自动注入 warnings 字段
_RUN_WARNINGS = []          # 显式告警（截断、诊断等），去重
_INF_SEEN = False           # 数据中出现过 inf（精确计数见 describe.n_inf / quality.non_finite）
_COMPLEX_EXAMPLES = []      # 复数字面量示例（最多留 3 个）
_REJECT_INF = {"on": False}  # --reject-inf 严格模式：inf 按缺失处理
_VERBOSITY = "normal"        # quiet / normal / verbose
_OUTPUT_FORMAT = "json"      # json / csv / md / html


def _warn(msg):
    """记录运行期告警，随下次 emit 以 warnings 字段输出（截断可见性 / 诊断提示）。"""
    if msg not in _RUN_WARNINGS:
        _RUN_WARNINGS.append(msg)


def _vlog(msg):
    """--verbose 模式下输出进度信息到 stderr（不影响 JSON 输出）。"""
    if _VERBOSITY == "verbose":
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def _dynamic_warnings():
    ws = list(_RUN_WARNINGS)
    if _INF_SEEN:
        ws.append("数据中存在非有限数值(inf)：%s；inf 会污染均值/方差等统计量"
                  "（各列精确计数见 describe 的 n_inf / quality 的 non_finite；可用 --reject-inf 按缺失处理）"
                  % ("已按 --reject-inf 严格模式作缺失处理" if _REJECT_INF["on"]
                     else "默认原值参与计算"))
    if _COMPLEX_EXAMPLES:
        ws.append("检测到复数形态字面量（如 %s）：内核不支持复数，已按缺失处理（该列可能被识别为分类列）"
                  % "、".join(_COMPLEX_EXAMPLES[:3]))
    return ws


def _col_signature(vals):
    """单列内容指纹：数值列用顺序无关聚合量（count/sum/sumsq/min/max），文本列用去重集合哈希。"""
    nums = []
    strs = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, bool):
            strs.append(str(v))
        elif isinstance(v, (int, float)):
            nums.append(float(v))
        else:
            strs.append(str(v))
    if nums:
        n = len(nums)
        s = sum(nums)
        sq = sum(x * x for x in nums)
        mn = min(nums)
        mx = max(nums)
        return "n%d:%.8f:%.8f:%.8f:%.8f" % (n, s, sq, mn, mx)
    hh = hashlib.sha256()
    uniq = sorted(set(strs))
    if len(uniq) <= 2000:
        hh.update("|".join(uniq).encode("utf-8"))
    else:
        for s in strs:
            hh.update(s.encode("utf-8"))
            hh.update(b"|")
    return "s%d:%s" % (len(strs), hh.hexdigest()[:24])


def _data_fingerprint(columns, rows):
    """数据集指纹（与行序无关、对内容敏感、O(n)）：各列签名的组合哈希。rows 为字典列表。"""
    h = hashlib.sha256()
    h.update(("%d" % len(rows)).encode("utf-8"))
    dict_rows = bool(rows) and isinstance(rows[0], dict)
    if not dict_rows:
        idx = {}
        for i, c in enumerate(columns):
            idx.setdefault(c, i)
    for c in sorted(columns):
        if dict_rows:
            vals = [r.get(c) for r in rows]
        else:
            ci = idx[c]
            vals = [r[ci] for r in rows if ci < len(r)]
        h.update(c.encode("utf-8"))
        h.update(b":")
        h.update(_col_signature(vals).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _compute_param_hash(args):
    """本次分析的方法学参数哈希（排除 input 路径与 func；数据身份由 data_fingerprint 承担）。"""
    d = {}
    for k, v in vars(args).items():
        if k in ("func", "input"):
            continue
        if v is None or v is False:
            continue
        d[k] = list(v) if isinstance(v, (list, tuple)) else v
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

# 资源上限：防止读取异常/恶意超大文件导致内存耗尽（H2 / M15）
_MAX_INPUT_BYTES = 200 * 1024 * 1024   # 单文件 200 MB
_MAX_INPUT_ROWS = 5_000_000            # 单表最多 500 万行


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in _MISSING_TOKENS


def _apply_cfg(args):
    if getattr(args, "alpha", None) is not None:
        a = float(args.alpha)
        if not (0.0 < a < 1.0):
            die("--alpha 必须介于 0 与 1 之间（不含端点），当前为 %r" % a)
        _CFG["alpha"] = a
    if getattr(args, "shapiro_max_n", None) is not None:
        _CFG["shapiro_max_n"] = int(args.shapiro_max_n)
    if getattr(args, "sentinels", None):
        try:
            _CFG["sentinels"] = [float(s) for s in str(args.sentinels).split(",") if s.strip()]
        except Exception:
            die("--sentinels 需为逗号分隔的数值，如 -999,-99,9999")


def to_float(x):
    global _INF_SEEN
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return None
        f = float(x)
    else:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in _MISSING_TOKENS:
            return None
        # E4：复数字面量（如 1+2j / 3j）显式拒绝并留痕，不再静默落入分类列
        if s[-1:] in ("j", "J") and any(ch.isdigit() for ch in s):
            if s not in _COMPLEX_EXAMPLES and len(_COMPLEX_EXAMPLES) < 3:
                _COMPLEX_EXAMPLES.append(s)
            return None
        try:
            f = float(s)
        except (ValueError, TypeError):
            return None
    # E1：inf 非有限值——默认原值放行但全局留痕（warnings 提示）；--reject-inf 时按缺失处理
    if math.isinf(f):
        _INF_SEEN = True
        return None if _REJECT_INF["on"] else f
    return f


def is_num(x):
    return to_float(x) is not None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def stdev(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def quantile(sorted_xs, q):
    n = len(sorted_xs)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_xs[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def pearson(x, y):
    xs = [to_float(a) for a in x]
    ys = [to_float(b) for b in y]
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return float("nan")
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    num = sum((a - mx) * (b - my) for a, b in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs))
    dy = math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def gaussian_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):

        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            return None
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x

def _gammp(a, x):
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(400):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-14:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)

    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 400):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def _betacf(a, b, x):
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betai(a, b, x):
    if a <= 0 or b <= 0:
        return float("nan")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    if a == 1.0:
        return 1.0 - (1.0 - x) ** b
    if b == 1.0:
        return x ** a
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _scipy_try(fn):
    if HAS_SCIPY:
        try:
            return fn()
        except Exception:
            pass
    return None


def _f_sf(F, df1, df2):
    sv = _scipy_try(lambda: float(spstats.f.sf(float(F), float(df1), float(df2))))
    if sv is not None:
        return sv
    if df1 <= 0 or df2 <= 0 or F <= 0:
        return 1.0 if F <= 0 else float("nan")
    x = df1 * F / (df1 * F + df2)
    return 1.0 - _betai(df1 / 2.0, df2 / 2.0, x)


def _chi2_sf(x, df):
    sv = _scipy_try(lambda: float(spstats.chi2.sf(float(x), float(df))))
    if sv is not None:
        return sv
    if x <= 0:
        return 1.0
    return 1.0 - _gammp(df / 2.0, x / 2.0)


def _t_two_sided_p(t, df):
    sv = _scipy_try(lambda: float(2.0 * spstats.t.sf(float(t), float(df))))
    if sv is not None:
        return sv
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def _t_ppf(q, df):
    if df is None or df != df or df <= 0:
        return None
    sv = _scipy_try(lambda: float(spstats.t.ppf(float(q), float(df))))
    if sv is not None:
        return sv
    target = 2.0 * (1.0 - q)
    lo, hi = 0.0, 1e3
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def _sig(p, alpha=None):
    a = _CFG["alpha"] if alpha is None else alpha
    return bool(p is not None and p == p and p < a)


def _fmt_p(p):
    if p is None or p != p:
        return "NA"
    return "<0.0001" if p < 0.0001 else ("%.4f" % p)


def _p_adjust(pvals, method):
    """对一组 p 值做多重比较校正。method: bonferroni / holm / fdr(BH)。"""
    m = len(pvals)
    if m == 0:
        return []
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pvals]
    if method == "holm":
        # Holm 逐步下降校正（按升序 p 控制族wise 错误率）
        order = sorted(range(m), key=lambda i: pvals[i])
        adj = [0.0] * m
        prev = 1.0
        for j in range(m - 1, -1, -1):
            i = order[j]
            prev = min(prev, pvals[i] * (m - j))
            adj[i] = min(1.0, prev)
        return adj
    if method == "fdr":
        # Benjamini-Hochberg 假发现率控制（按升序秩 i 校正 p_i*m/i，自大向小累计取最小）
        order = sorted(range(m), key=lambda i: pvals[i])
        adj = [0.0] * m
        prev = 1.0
        for j in range(m - 1, -1, -1):
            i = order[j]
            prev = min(prev, pvals[i] * m / (j + 1))
            adj[i] = prev
        return [min(1.0, adj[i]) for i in range(m)]
    return list(pvals)


def _mean_ci(xs, level=0.95):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m, s = mean(xs), stdev(xs)
    tc = _t_ppf(0.5 + level / 2.0, n - 1)
    if tc is None or s != s:
        return None
    h = tc * s / math.sqrt(n)
    return [round(m - h, 4), round(m + h, 4)]


def _norm_cdf(z):
    """标准正态累积分布（零依赖路径用误差函数近似）。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(q):
    """标准正态分位数（逆 CDF）。优先 scipy，零依赖用 Acklam 有理近似。"""
    if q is None or q != q or q <= 0.0 or q >= 1.0:
        return None
    sv = _scipy_try(lambda: float(spstats.norm.ppf(float(q))))
    if sv is not None:
        return sv
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if q < plow:
        xx = math.sqrt(-2.0 * math.log(q))
        return (((((c[0]*xx+c[1])*xx+c[2])*xx+c[3])*xx+c[4])*xx+c[5]) / \
               ((((d[0]*xx+d[1])*xx+d[2])*xx+d[3])*xx+1.0)
    if q > phigh:
        xx = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0]*xx+c[1])*xx+c[2])*xx+c[3])*xx+c[4])*xx+c[5]) / \
                ((((d[0]*xx+d[1])*xx+d[2])*xx+d[3])*xx+1.0)
    xx = q - 0.5
    r = xx * xx
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*xx / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _norm_sf(z):
    return 1.0 - _norm_cdf(z)


def _fisher_z_ci(r, n, level=0.95):
    """相关系数 r 的 Fisher z 变换置信区间（零依赖可用）。"""
    if n is None or n < 4 or r != r or abs(r) >= 1.0:
        return None
    zr = 0.5 * math.log((1.0 + r) / (1.0 - r))
    se = 1.0 / math.sqrt(n - 3)
    zc = _norm_ppf(0.5 + level / 2.0)
    if zc is None:
        return None
    lo = zr - zc * se
    hi = zr + zc * se
    return [round((math.exp(2*lo)-1)/(math.exp(2*lo)+1), 4),
            round((math.exp(2*hi)-1)/(math.exp(2*hi)+1), 4)]


def _mann_whitney_u(a, b):
    """两独立样本 Mann-Whitney U 检验（零依赖正态近似 + 结校正）。返回 (U1, p_two_sided)。"""
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        return None, None
    combined = a + b
    rk = _ranks(combined)
    r1 = sum(rk[i] for i in range(n1))
    U1 = n1 * n2 + n1 * (n1 + 1) / 2.0 - r1
    mu = n1 * n2 / 2.0
    ties = Counter(combined)
    tcorr = sum(t**3 - t for t in ties.values() if t > 1)
    denom = (n1 * n2 * (n1 + n2 + 1)) / 12.0
    if tcorr and denom > 0:
        denom *= 1.0 - tcorr / ((n1 + n2) ** 3 - (n1 + n2))
    if denom <= 0:
        return U1, None
    z = (U1 - mu) / math.sqrt(denom)
    p = _norm_sf(abs(z)) * 2.0
    return U1, min(1.0, p)


def _cut_label(v, cuts):
    """阈值映射：|v| 落入首个 <cut 区间返回对应标签，否则默认「大」。"""
    if v is None or v != v:
        return None
    v = abs(v)
    for cut, name in cuts:
        if v < cut:
            return name
    return "大"


def _cliff_label(d):
    """Cliff's delta 解释阈值（Romano et al. 2006，按 |δ|）：
    <0.147 可忽略；0.147–0.33 小；0.33–0.474 中；≥0.474 大。"""
    return _cut_label(d, ((0.147, "可忽略"), (0.33, "小"), (0.474, "中")))


def _nonparam_effect(a, b, u1=None):
    """两独立样本非参效应量。返回 (cliff_delta, rank_biserial) 或 None。
    Cliff's delta = P(a>b) - P(a<b) = (2U_a - n1 n2)/(n1 n2)，范围 [-1, 1]；
    rank_biserial (Kerby 2014) = (P(a>b)-P(a<b))/(P(a>b)+P(a<b))
        = (2U_a - n1 n2)/(n1 n2 - e)，e 为跨组相等对数（剔除结后再标准化）。
    注意：_mann_whitney_u(a,b) 实际返回的是 U_b = n1 n2 - U_a，故这里用
    u_a = denom - u1 还原 a 组的 U，确保符号约定为「a 整体更大则 δ>0」。"""
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        return None
    if u1 is None:
        u1, _ = _mann_whitney_u(a, b)
        if u1 is None:
            return None
    denom = n1 * n2
    if denom <= 0:
        return None
    ua = denom - u1
    delta = (2.0 * ua - denom) / denom
    bc = Counter(b)
    eq = sum(bc.get(x, 0) for x in a)
    fplusu = denom - eq
    rb = (2.0 * ua - denom) / fplusu if fplusu > 0 else 0.0
    return delta, rb


def _kw_posthoc(groups, alpha=None):
    """Kruskal-Wallis 事后两两比较：Mann-Whitney U + Benjamini-Hochberg 校正 + 非参效应量。"""
    a = alpha or _CFG["alpha"]
    levels = list(groups.keys())
    pairs, raw, us, effs = [], [], [], []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            ga, gb = groups[levels[i]], groups[levels[j]]
            u, p = _mann_whitney_u(ga, gb)
            pairs.append((levels[i], levels[j])); raw.append(p)
            us.append(round(u, 1)); effs.append(_nonparam_effect(ga, gb, u1=u))
    out = []
    if raw:
        valid = [p for p in raw if p is not None]
        adj_all = _p_adjust(valid, "fdr") if valid else []
        ai = 0
        for k, (lv1, lv2) in enumerate(pairs):
            p = raw[k]
            ap = adj_all[ai] if (p is not None and ai < len(adj_all)) else None
            if p is not None:
                ai += 1
            d, rb = (effs[k] or (None, None))
            out.append({"a": lv1, "b": lv2, "U": us[k],
                        "p_value": _r4(p), "adjusted_p": _r4(ap),
                        "cliff_delta": _r4(d), "rank_biserial": _r4(rb),
                        "effect_size_label": _cliff_label(d),
                        "significant": bool(p is not None and p < a and (ap is None or ap < a))})
    return out


def _ratio_2x2(m, kind):
    """m=[[a,b],[c,d]]；kind ∈ {or, rr}。返回 (比值, [lo,hi], valid)，Woolf 对数 CI。
    OR=(ad)/(bc)；RR=(a/(a+b))/(c/(c+d))。"""
    a, b = m[0][0], m[0][1]
    c, d = m[1][0], m[1][1]
    if kind == "or":
        if a <= 0 or b <= 0 or c <= 0 or d <= 0:
            return None, None, False
        val = (a * d) / (b * c)
        se = math.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
    else:
        n1, n2 = a + b, c + d
        if n1 <= 0 or n2 <= 0:
            return None, None, False
        p1, p2 = a / n1, c / n2
        if p1 <= 0 or p2 <= 0:
            return None, None, False
        val = p1 / p2
        se = math.sqrt(1.0/a - 1.0/n1 + 1.0/c - 1.0/n2)
    zc = _norm_ppf(0.975)
    if zc is None:
        return round(val, 4), None, True
    return round(val, 4), [round(math.exp(math.log(val) - zc * se), 4),
                           round(math.exp(math.log(val) + zc * se), 4)], True


def _odds_ratio_2x2(m):
    return _ratio_2x2(m, "or")


def _risk_ratio_2x2(m):
    return _ratio_2x2(m, "rr")


def _cohen_d_ci(d, n1, n2, level=0.95):
    """Cohen's d / Glass's delta 的 Hedges-Olkin 近似置信区间。"""
    if d is None or n1 < 2 or n2 < 2:
        return None
    se = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2 - 2)))
    zc = _norm_ppf(0.5 + level / 2.0)
    if zc is None:
        return None
    return [round(d - zc * se, 4), round(d + zc * se, 4)]


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _group_values(rows, factor, value):
    groups = OrderedDict()
    for r in rows:
        y = to_float(r.get(value))
        if y is not None:
            groups.setdefault(str(r.get(factor)), []).append(y)
    return groups


def _numeric_columns(columns, rows):
    return [c for c in columns if any(is_num(r.get(c)) for r in rows)]


def _col_floats(rows, c, dropna=False):
    xs = [to_float(r.get(c)) for r in rows]
    if dropna:
        xs = [x for x in xs if x is not None]
    return xs


def _numeric_series(rows, num):
    return {c: _col_floats(rows, c) for c in num}


def _atomic_write_core(absp, encoding, write_fn):
    """原子写骨架：建父目录 → 唯一临时文件 → 写入 → os.replace → chmod → 异常清理。
    临时文件名含 PID+随机串（tempfile），规避同 PID 并发写同目标路径的竞态（H2）。"""
    parent = os.path.dirname(absp)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline="",
                                         dir=parent or ".",
                                         prefix=os.path.basename(absp) + ".tmp.",
                                         suffix=".tmp", delete=False) as f:
            tmp = f.name
            write_fn(f)
        os.replace(tmp, absp)
        try:
            os.chmod(absp, 0o644)
        except Exception:
            pass
        tmp = None
    except Exception:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise


def _atomic_write(path, text, encoding="utf-8-sig"):
    """确保父目录存在并以原子方式写入（先写唯一临时文件再 os.replace），避免半写或覆盖损坏。"""
    _atomic_write_core(os.path.abspath(path), encoding, lambda f: f.write(text))


def _write_rows_csv(path, rows, fieldnames):
    def w(f):
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in fieldnames})
    _atomic_write_core(os.path.abspath(path), "utf-8-sig", w)


def _effect_label(kind, v):
    cuts = {"d": ((0.2, "极小"), (0.5, "小"), (0.8, "中等")),
            "eta2": ((0.01, "极小"), (0.06, "小"), (0.14, "中等")),
            "cramer_v": ((0.1, "极小"), (0.3, "小"), (0.5, "中等"))}[kind]
    return _cut_label(v, cuts)

def _detect_csv_encoding(path):
    """尝试 utf-8-sig → gbk → gb18030，返回可成功解码的编码名；全部失败则回退 utf-8-sig（errors=replace）。"""
    for enc in ("utf-8-sig", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(8192)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8-sig"

def load_rows(path, fmt="auto", max_rows=None):
    if not os.path.exists(path):
        die("文件不存在: %s" % path)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    if size > _MAX_INPUT_BYTES:
        die("文件过大（%d 字节 > 单文件上限 %d 字节），已拒绝读取以防资源耗尽" % (size, _MAX_INPUT_BYTES))
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = {".csv": "csv", ".tsv": "tsv", ".txt": "tsv",
               ".json": "json", ".jsonl": "json", ".xlsx": "xlsx"}.get(ext, "csv")
    cap = max_rows if max_rows is not None else _MAX_INPUT_ROWS
    rows = []
    columns = []
    if fmt in ("csv", "tsv"):
        delim = "," if fmt == "csv" else "\t"
        # 大文件加速：pandas 解析（dtype=str 保留原始字符串值，行为等同 csv.DictReader），失败回退标准库
        if HAS_PANDAS and size > 5 * 1024 * 1024:
            try:
                _enc = _detect_csv_encoding(path)
                df = pd.read_csv(path, sep=delim, dtype=str, keep_default_na=False,
                                 nrows=cap, encoding=_enc)
                if _enc != "utf-8-sig":
                    _warn("CSV 非 UTF-8 编码，已以 %s 解码读取（建议保存为 UTF-8-BOM 以避免兼容问题）" % _enc)
                columns = [str(c) for c in df.columns]
                rows = df.to_dict("records")
                _PROV["data_fingerprint"] = _data_fingerprint(columns, rows)
                if len(rows) >= cap:
                    _warn("已读取 %d 行（达到行数上限，文件可能被截断；可用 --max-rows 显式设定）" % len(rows))
                return columns, rows
            except Exception:
                pass  # 回退标准库
        truncated = False
        _enc = _detect_csv_encoding(path)
        try:
            with open(path, "r", encoding=_enc, newline="") as f:
                _csv_text = f.read()
            if _enc != "utf-8-sig":
                _warn("CSV 非 UTF-8 编码，已以 %s 解码读取（建议保存为 UTF-8-BOM 以避免兼容问题）" % _enc)
        except (UnicodeDecodeError, LookupError):
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                _csv_text = f.read()
        reader = csv.DictReader(io.StringIO(_csv_text), delimiter=delim)
        columns = list(reader.fieldnames or [])
        for i, row in enumerate(reader):
            if i >= cap:
                truncated = True
                break
            rows.append(dict(row))
        if truncated:
            _warn("文件超过 %d 行上限，仅读取前 %d 行（结果基于截断数据）" % (cap, len(rows)))
    elif fmt == "json":
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                data = json.load(f)
        except Exception as e:
            die("JSON 解析失败: %s（%s）" % (path, e))
        if isinstance(data, dict):
            if "rows" in data and isinstance(data["rows"], list):
                data = data["rows"]
            elif data and all(isinstance(v, list) for v in data.values()):
                cols = list(data.keys())
                n = max(len(v) for v in data.values())
                rows = [{c: (data[c][i] if i < len(data[c]) else None) for c in cols}
                        for i in range(n)]
                columns = cols
                _PROV["data_fingerprint"] = _data_fingerprint(columns, rows)
                return columns, rows
            else:
                data = [data]
        if isinstance(data, list):
            truncated = False
            for i, item in enumerate(data):
                if i >= cap:
                    truncated = True
                    break
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    rows.append({"value": item})
            if truncated:
                _warn("JSON 数组超过 %d 行上限，仅读取前 %d 项（结果基于截断数据）" % (cap, len(rows)))
            if rows:
                columns = list(rows[0].keys())
    elif fmt == "xlsx":
        if HAS_PANDAS:
            try:
                df = pd.read_excel(path)
            except Exception as e:
                die("读取 xlsx 失败: %s" % e)
            if len(df) > _MAX_INPUT_ROWS:
                die("xlsx 行数过多（%d > 上限 %d），已拒绝读取" % (len(df), _MAX_INPUT_ROWS))
            columns = list(df.columns)
            rows = df.where(df.notna(), None).to_dict(orient="records")
        else:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                ws = wb.active
                data = list(ws.iter_rows(values_only=True, max_row=_MAX_INPUT_ROWS + 2))
                if len(data) > _MAX_INPUT_ROWS + 1:
                    die("xlsx 行数过多（> %d），已拒绝读取" % _MAX_INPUT_ROWS)
                if data:
                    columns = [str(c) if c is not None else "col%d" % i
                               for i, c in enumerate(data[0])]
                    for r in data[1:]:
                        rows.append({columns[i]: (r[i] if i < len(r) else None)
                                     for i in range(len(columns))})
            except Exception as e:
                die("读取 xlsx 需要 openpyxl 或 pandas：%s" % e)
    else:
        die("不支持的格式: %s" % fmt)
    _PROV["data_fingerprint"] = _data_fingerprint(columns, rows)
    return columns, rows

def cmd_load(args):
    columns, rows = load_rows(args.input, args.format, args.max_rows)
    numeric_cols = [c for c in columns if any(is_num(r.get(c)) for r in rows[: min(50, len(rows))])]
    sample = rows[: min(5, len(rows))]
    emit({
        "status": "ok", "task": "load",
        "result": {
            "format": args.format,
            "total_rows": len(rows),
            "columns": columns,
            "numeric_columns": numeric_cols,
            "dtypes": {c: ("numeric" if c in numeric_cols else "text") for c in columns},
            "preview": sample,
        },
    })

def cmd_clean(args):
    columns, rows = load_rows(args.input)
    n_in = len(rows)

    if args.drop_empty_cols:
        nonempty = [c for c in columns if any(not _is_missing(r.get(c)) for r in rows)]
        rows = [{c: r.get(c) for c in nonempty} for r in rows]
        columns = nonempty

    numeric_cols = _numeric_columns(columns, rows)
    for r in rows:
        for c in columns:
            v = r.get(c)
            if c in numeric_cols:
                r[c] = to_float(v)
            else:
                if _is_missing(v):
                    r[c] = None
                else:
                    r[c] = str(v).strip()

    missing_before = sum(1 for r in rows for c in numeric_cols if r.get(c) is None)
    if args.missing == "drop":
        rows = [r for r in rows if all(r.get(c) is not None for c in numeric_cols)]
    elif args.missing in ("mean", "median"):
        for c in numeric_cols:
            vals = [r[c] for r in rows if r.get(c) is not None]
            fill = mean(vals) if args.missing == "mean" else median(vals)
            for r in rows:
                if r.get(c) is None:
                    r[c] = fill
    elif args.missing == "ffill":
        for c in numeric_cols:
            last = None
            for r in rows:
                if r.get(c) is None:
                    r[c] = last
                else:
                    last = r[c]
    missing_after = sum(1 for r in rows for c in numeric_cols if r.get(c) is None)

    seen = set()
    dedup = []
    dup = 0
    for r in rows:
        key = tuple((c, r.get(c)) for c in columns)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        dedup.append(r)
    rows = dedup

    if args.output:
        _write_rows_csv(args.output, rows, columns)
    emit({
        "status": "ok", "task": "clean",
        "result": {
            "rows_in": n_in,
            "rows_out": len(rows),
            "duplicates_removed": dup,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "missing_strategy": args.missing,
            "numeric_columns": numeric_cols,
            "output": args.output or None,
            "columns": columns,
        },
    })

def _iqr_bounds(xs, k=1.5):
    xs = sorted(x for x in xs if x is not None)
    q1 = quantile(xs, 0.25)
    q3 = quantile(xs, 0.75)
    iqr = q3 - q1
    return q1, q3, q1 - k * iqr, q3 + k * iqr


def _anomaly_column(vals, method, grade, contamination=None):
    idx = [i for i, v in enumerate(vals) if v is not None]
    xs = [vals[i] for i in idx]
    n = len(xs)
    if n < 4:
        return {"indices": [], "count": 0, "bounds": None, "note": "样本不足"}

    k_iqr = {"L1": 1.5, "L2": 3.0, "L3": 5.0}[grade]
    t_z = {"L1": 2.0, "L2": 3.0, "L3": 4.0}[grade]
    eff = method
    if method == "isolation_forest" and not HAS_SKLEARN:
        eff = "iqr"
    if method == "mahalanobis" and not (HAS_SCIPY and HAS_NUMPY):
        eff = "iqr"
    definition = None
    threshold = None
    if eff == "iqr":
        _, _, lo, hi = _iqr_bounds(xs, k_iqr)
        flags = [idx[i] for i, v in enumerate(xs) if v < lo or v > hi]
        used = "iqr"
        threshold = k_iqr
        definition = "x < Q1-%.1f*IQR 或 x > Q3+%.1f*IQR" % (k_iqr, k_iqr)
    elif eff == "zscore":
        m, s = mean(xs), stdev(xs)
        if s == 0:
            return {"indices": [], "count": 0, "bounds": None, "note": "方差为0"}
        lo, hi = m - t_z * s, m + t_z * s
        flags = [idx[i] for i, v in enumerate(xs) if abs(v - m) > t_z * s]
        used = "zscore"
        threshold = t_z
        definition = "|x-mean|/sd > %.1f" % t_z
    elif eff == "mad":
        med = median(xs)
        absdev = [abs(v - med) for v in xs]
        mad = median(absdev) or 1e-9
        scale = mad / _CFG["mad_scale"]
        lo, hi = med - t_z * scale, med + t_z * scale
        flags = [idx[i] for i, v in enumerate(xs) if abs(v - med) > t_z * scale]
        used = "mad"
        threshold = t_z
        definition = "|%.4f*(x-med)/MAD| > %.1f（modified z-score，与 zscore 的 σ 倍数同义）" % (
            _CFG["mad_scale"], t_z)
    elif eff == "isolation_forest":
        arr = np.array([[v] for v in xs])
        cont = contamination if contamination is not None else _CFG["if_contamination"][grade]
        clf = IsolationForest(contamination=cont, random_state=0)
        pred = clf.fit_predict(arr)
        flags = [idx[i] for i, p in enumerate(pred) if p == -1]
        inl = [xs[i] for i, p in enumerate(pred) if p == 1]
        lo = min(inl) if inl else min(xs)
        hi = max(inl) if inl else max(xs)
        used = "isolation_forest"
        threshold = cont
        definition = "contamination=%.3f（预期异常占比，随 grade 递减与其他方法同向）" % cont
    elif eff == "mahalanobis":
        arr = np.array(xs, dtype=float)
        mu = float(arr.mean())
        cov = float(np.cov(arr))
        cov = cov if cov != 0 else 1e-9
        sd = math.sqrt(cov)
        lo, hi = mu - t_z * sd, mu + t_z * sd
        d = np.abs(arr - mu) / sd
        flags = [idx[i] for i, v in enumerate(d) if v > t_z]
        used = "mahalanobis"
        threshold = t_z
        definition = "标准化距离 |x-mean|/sd > %.1f（一维马氏距离退化形式，与 zscore 同义）" % t_z
    else:
        return {"indices": [], "count": 0, "bounds": None, "note": "未知方法"}
    return {"indices": flags, "count": len(flags),
            "bounds": [lo, hi] if lo is not None else None, "used": used,
            "threshold": threshold, "definition": definition}


def _is_identifier_column(columns, rows, c):
    vals = _col_floats(rows, c, dropna=True)
    if len(vals) < 2:
        return False
    if len(set(vals)) != len(vals):
        return False
    if not all(float(v).is_integer() for v in vals):
        return False
    s = sorted(vals)
    consecutive = all(s[i] == s[0] + i for i in range(len(s)))
    name_id = any(k in c.lower() for k in ("id", "编号", "序号", "index", "样本号", "sample"))
    return bool(consecutive or name_id)


def _discipline_anomaly(series, disc, c, rules_used):
    rule = _discipline_rule_for_column(disc, c)
    method = rule["method"]
    grade = rule.get("grade") or _grade_for_rule(method, rule.get("factor"), rule.get("contamination"))
    cont = rule.get("contamination") if method == "isolation_forest" else None
    r = _anomaly_column(series, method, grade, contamination=cont)
    r["discipline_rule"] = rule["id"]
    r["advice"] = rule.get("advice", "")
    r["grade"] = grade
    rules_used.add(rule["id"])
    return r


def cmd_anomaly(args):
    columns, rows = load_rows(args.input)
    numeric_cols = _numeric_columns(columns, rows)
    if args.field:
        target = [c.strip() for c in args.field.split(",") if c.strip() in numeric_cols]
        if not target:
            die("指定字段不存在或非数值: %s" % args.field)
        skipped = [c for c in numeric_cols if c not in target]
    else:

        target = [c for c in numeric_cols if not _is_identifier_column(columns, rows, c)]
        skipped = [c for c in numeric_cols if c not in target]

    disc = None
    if getattr(args, "discipline", None):
        disc = _discipline_index(args.discipline)
        if disc == "auto":
            disc = _discipline_index(infer_discipline(columns)["key"])
        elif disc is None:
            die("未知学科: %s（可选: %s / auto）" % (
                args.discipline, ", ".join(d["key"] for d in DISCIPLINE_REGISTRY)))
    report = OrderedDict()
    repaired = None
    if args.repair:
        repaired = [{c: r.get(c) for c in columns} for r in rows]
    rules_used = set()
    for c in target:
        series = [to_float(r.get(c)) for r in rows]
        if disc:
            res = _discipline_anomaly(series, disc, c, rules_used)
        else:
            res = _anomaly_column(series, args.method, args.grade,
                                  contamination=getattr(args, "contamination", None))
        report[c] = res
        if args.repair and res.get("bounds"):
            lo, hi = res["bounds"]
            for i, r in enumerate(repaired):
                v = to_float(r.get(c))
                if v is not None and (v < lo or v > hi):
                    r[c] = hi if v > hi else lo
    total = sum(v["count"] for v in report.values())
    if args.repair and args.output:
        _write_rows_csv(args.output, repaired, columns)
    if disc:

        used_methods = sorted(set(v.get("used") for v in report.values() if v.get("used")))
        method_str = ",".join(used_methods) if used_methods else args.method
        grades = sorted(set(v.get("grade") for v in report.values() if v.get("grade")))
        grade_str = ",".join(grades) if grades else None
    else:
        method_str = args.method
        grade_str = args.grade
    meta = {
        "method": method_str,
        "grade": grade_str,
        "numeric_columns": numeric_cols,
        "analyzed_columns": target,
        "excluded_columns": skipped,
        "total_anomalies": total,
        "per_column": report,
        "repaired_output": args.output if args.repair else None,
    }
    if disc:
        meta["discipline"] = {"key": disc["key"], "name": disc["name"],
                              "rules_applied": sorted(rules_used)}
    emit({"status": "ok", "task": "anomaly", "result": meta})

_DISCIPLINE_REGISTRY_EMBEDDED = [
    {
        "key": "biomed", "name": "生物医学 / 基础医学 / 临床医学", "alias": ["生物"],
        "keywords": ["log2fc", "neglog10p", "基因", "表达量", "随访时间", "事件",
                     "go_term", "通路", "m_logratio", "a_mean", "效应量", "组别", "系列"],
        "directions": [
            {"name": "高通量组学", "keywords": ["log2fc", "neglog10p", "go_term", "通路", "m_logratio", "a_mean", "表达量"]},
            {"name": "单细胞测序 scRNA-seq", "keywords": ["dim1", "dim2", "基因", "表达量", "系列"]},
            {"name": "细胞 / 动物实验", "keywords": ["参数1", "参数2", "分组", "系列"]},
            {"name": "临床流行病学 / 病例分析", "keywords": ["随访时间", "事件", "效应量", "ci宽度", "研究"]},
            {"name": "病理 / 影像学", "keywords": ["权重", "通路靶"]},
        ],
        "rules": [
            {"id": "biomed_omics_batch", "method": "isolation_forest", "contamination": 0.05, "grade": "L2",
             "advice": "优先用 ComBat/limma 做批次校正并核对批次变量；具有明确生物意义（如驱动突变）的离群须保留。",
             "keywords": ["log2fc", "neglog10p", "表达量", "go_term", "通路"], "applicable_fields": ["log2fc", "表达量", "a_mean"]},
            {"id": "biomed_scrna_droplet", "method": "isolation_forest", "contamination": 0.08, "grade": "L2",
             "advice": "按线粒体比例/UMI 数过滤空滴；用 scrublet 识别 doublet；真实稀有细胞类型应保留。",
             "keywords": ["dim1", "dim2", "基因", "表达量"], "applicable_fields": ["dim1", "dim2", "表达量"]},
            {"id": "biomed_clinical_extreme", "method": "iqr", "factor": 3.0, "grade": "L2",
             "advice": "核查是否删失/录入错误；极端但真实者保留并做敏感性分析（如 Cox 稳健估计）。",
             "keywords": ["随访时间", "事件", "效应量", "ci宽度"], "applicable_fields": ["随访时间", "效应量"]},
        ],
        "viz": ["volcano", "ma", "heatmap", "bubble", "sankey", "umap", "ridge", "errorbar",
                "line", "box", "flow", "mosaic", "km", "forest", "roc"],
    },
    {
        "key": "agri_env", "name": "农林环境科学（土壤/植物/微生物/生态）", "alias": ["环境"],
        "keywords": ["养分mg", "污染物mg", "土层", "土地利用", "样点", "物种", "门水平", "相对丰度",
                     "株高cm", "生物量g", "光合速率", "降雨mm", "流域", "生境", "施肥梯度", "湿度", "温度"],
        "directions": [
            {"name": "土壤理化 / 环境监测", "keywords": ["养分mg", "污染物mg", "土层", "土地利用", "样点", "湿度", "温度"]},
            {"name": "植物生理 / 栽培实验", "keywords": ["株高cm", "生物量g", "光合速率", "植物", "施肥梯度"]},
            {"name": "微生物组", "keywords": ["门水平", "相对丰度", "物种"]},
            {"name": "生态 / 污染修复 / 流域", "keywords": ["流域", "生境", "降雨mm", "流量权重"]},
        ],
        "rules": [
            {"id": "agri_env_timeseries", "method": "iqr", "factor": 3.0, "grade": "L2",
             "advice": "对孤立跳点优先插值修复或标注，避免直接删除以丢失时序连续性。",
             "keywords": ["养分mg", "污染物mg", "温度"], "applicable_fields": ["养分mg", "污染物mg", "温度"]},
            {"id": "agri_env_community", "method": "isolation_forest", "contamination": 0.05, "grade": "L2",
             "advice": "核查物种注释与测序深度；稀有但真实的分类单元（如关键功能菌）应保留。",
             "keywords": ["门水平", "相对丰度", "物种"], "applicable_fields": ["相对丰度", "物种"]},
        ],
        "viz": ["line", "kriging", "contour", "box", "grouped", "heatmap", "sankey", "stacked",
                "umap", "ridge", "bubble", "mosaic"],
    },
    {
        "key": "materials", "name": "材料科学 / 高分子 / 金属 / 新能源", "alias": ["材料"],
        "keywords": ["应力mpa", "应变", "电流ma", "电压v", "容量mah", "阻抗ohm", "粒径nm",
                     "角度deg", "波长nm", "催化活性", "材料类型", "元素占比", "温度k"],
        "directions": [
            {"name": "材料形貌 / 微观结构表征", "keywords": ["粒径nm", "角度deg", "波长nm", "强度", "元素"]},
            {"name": "力学 / 电化学 / 光电性能", "keywords": ["应力mpa", "应变", "电流ma", "电压v", "容量mah", "阻抗ohm", "温度k"]},
            {"name": "有限元仿真 / 数值模拟", "keywords": ["温度k", "时间s", "材料类型"]},
            {"name": "催化 / 吸附 / 储能机理", "keywords": ["催化活性", "元素", "元素占比"]},
        ],
        "rules": [
            {"id": "mat_mech_overlimit", "method": "iqr", "factor": 1.5, "grade": "L3",
             "advice": "超量程/超规格读数视为无效，检查夹具与传感器；材料退化拐点（如容量跳水）须保留。",
             "keywords": ["应力mpa", "电压v", "容量mah"], "applicable_fields": ["应力mpa", "电压v", "容量mah"]},
            {"id": "mat_morphology_outlier", "method": "isolation_forest", "contamination": 0.06, "grade": "L2",
             "advice": "核对标样与成像参数；团聚体须区别于真实纳米颗粒，必要时复测。",
             "keywords": ["粒径nm", "强度"], "applicable_fields": ["粒径nm", "强度"]},
        ],
        "viz": ["hist", "line", "errorbar", "contour", "network", "stacked", "volcano"],
    },
    {
        "key": "cs", "name": "计算机 / 人工智能 / 机器学习", "alias": ["计算机"],
        "keywords": ["准确率", "精确率", "召回率", "f1", "损失", "验证损失", "轮次", "特征", "重要性",
                     "维度1", "维度2", "注意力", "参数量m", "速度ms", "任务", "模型"],
        "directions": [
            {"name": "模型性能评估", "keywords": ["准确率", "精确率", "召回率", "f1", "损失", "验证损失", "轮次"]},
            {"name": "数据降维 / 特征分析", "keywords": ["维度1", "维度2", "特征", "重要性", "类别"]},
            {"name": "深度学习 / 图像时序任务", "keywords": ["注意力", "速度ms", "参数量m", "模型"]},
            {"name": "通信 / 微电子 / 信号处理", "keywords": ["频谱", "信号", "信噪"]},
        ],
        "rules": [
            {"id": "cs_metric_crash", "method": "iqr", "factor": 3.0, "grade": "L3",
             "advice": "NaN/Inf 多因除零或梯度溢出，检查学习率与数值稳定；突然 plateau 多为学习率不当。",
             "keywords": ["准确率", "损失", "验证损失"], "applicable_fields": ["准确率", "损失", "验证损失"]},
            {"id": "cs_dim_outlier", "method": "isolation_forest", "contamination": 0.05, "grade": "L2",
             "advice": "核查是否为标注错误/对抗样本；边界但合理的簇应保留。",
             "keywords": ["维度1", "维度2"], "applicable_fields": ["维度1", "维度2"]},
        ],
        "viz": ["roc", "heatmap", "grouped", "line", "umap", "scatter", "network", "contour"],
    },
    {
        "key": "chem", "name": "化学 / 化工 / 分析化学 / 催化", "alias": ["化学"],
        "keywords": ["波长nm", "吸光度", "化学位移ppm", "强度", "时间min", "浓度", "反应速率",
                     "产率", "催化剂", "反应物", "产物", "谱峰编号", "反应器x", "反应器y"],
        "directions": [
            {"name": "光谱 / 色谱分析", "keywords": ["波长nm", "吸光度", "化学位移ppm", "强度", "谱峰编号"]},
            {"name": "反应动力学 / 热力学", "keywords": ["时间min", "浓度", "反应速率", "温度", "产率"]},
            {"name": "化工流程 / 物质转化", "keywords": ["催化剂", "反应物", "产物", "流量权重", "反应器x", "反应器y"]},
        ],
        "rules": [
            {"id": "chem_spectrum_noise", "method": "zscore", "factor": 3.0, "grade": "L2",
             "advice": "仪器尖峰建议平滑或剔除；若为依据明确的真实吸收峰须保留。",
             "keywords": ["吸光度", "强度", "化学位移ppm"], "applicable_fields": ["吸光度", "强度", "化学位移ppm"]},
            {"id": "chem_reaction_overlimit", "method": "iqr", "factor": 1.5, "grade": "L3",
             "advice": "负浓度/超 100% 产率为计算或录入错误；动力学拐点（如自加速）须保留。",
             "keywords": ["浓度", "产率", "反应速率"], "applicable_fields": ["浓度", "产率", "反应速率"]},
        ],
        "viz": ["line", "errorbar", "sankey", "stacked", "contour"],
    },
    {
        "key": "civil_geo", "name": "土木水利 / 地质资源 / 测绘地理", "alias": ["土木地质", "土木"],
        "keywords": ["荷载kn", "位移mm", "强度mpa", "试样", "降雨量mm", "高程m", "流域", "流量权重",
                     "地层", "矿物占比", "x坐标", "y坐标", "地球化学指标", "构件", "指标值"],
        "directions": [
            {"name": "岩土 / 结构力学仿真", "keywords": ["荷载kn", "位移mm", "强度mpa", "试样", "构件"]},
            {"name": "水文 / 流域 / 水资源", "keywords": ["降雨量mm", "高程m", "流域", "流量权重"]},
            {"name": "地质勘探 / 矿产资源", "keywords": ["地层", "矿物占比", "地球化学指标", "元素", "x坐标", "y坐标"]},
        ],
        "rules": [
            {"id": "civil_geo_load_overlimit", "method": "iqr", "factor": 1.5, "grade": "L3",
             "advice": "超设计值核查加载与传感器；灾变前兆（位移突变）必须保留并预警。",
             "keywords": ["荷载kn", "位移mm", "强度mpa"], "applicable_fields": ["荷载kn", "位移mm", "强度mpa"]},
            {"id": "civil_geo_geochem", "method": "isolation_forest", "contamination": 0.05, "grade": "L2",
             "advice": "区分仪器漂移与真实矿化异常；高值靶区须复核取样与测试方法。",
             "keywords": ["矿物占比", "地球化学指标"], "applicable_fields": ["矿物占比", "地球化学指标"]},
        ],
        "viz": ["contour", "line", "box", "sankey", "stacked", "scatter"],
    },
    {
        "key": "econ_soc", "name": "经济学 / 管理学 / 社科（心理/社会）", "alias": ["经济社科", "经济"],
        "keywords": ["满意度", "得分", "学历", "收入分层", "年份", "gdp", "物价", "自变量",
                     "因变量", "效应量", "ci宽度", "行为选择", "政策群体", "网络节点", "节点度", "人群"],
        "directions": [
            {"name": "问卷调研 / 统计描述", "keywords": ["满意度", "得分", "学历", "收入分层", "人群", "性别"]},
            {"name": "计量经济学 / 回归分析", "keywords": ["自变量", "因变量", "效应量", "ci宽度", "年份", "gdp", "物价", "研究"]},
            {"name": "行为心理 / 社会网络", "keywords": ["行为选择", "政策群体", "网络节点", "节点度", "类别"]},
        ],
        "rules": [
            {"id": "econ_survey_outlier", "method": "iqr", "factor": 3.0, "grade": "L2",
             "advice": "矛盾题/直线作答视为无效；天花板效应须报告而非删除，并在分析中注明。",
             "keywords": ["满意度", "得分"], "applicable_fields": ["满意度", "得分"]},
            {"id": "econ_reg_influential", "method": "iqr", "factor": 3.0, "grade": "L2",
             "advice": "用 Cook 距离识别强影响点；确认后做稳健回归或敏感性分析，勿盲目删除。",
             "keywords": ["因变量", "效应量"], "applicable_fields": ["因变量", "效应量"]},
        ],
        "viz": ["grouped", "mosaic", "box", "scatter", "forest", "line", "heatmap", "sankey", "network"],
    },
    {
        "key": "math_stat", "name": "数学 / 统计学", "alias": ["数学"],
        "keywords": ["数值x", "数值y", "维度1", "维度2", "相关系数", "拟合x", "拟合y", "样本", "组别", "计数"],
        "directions": [],
        "rules": [
            {"id": "math_stat_outlier", "method": "iqr", "factor": 1.5, "grade": "L1",
             "advice": "按研究目的决定留删；小样本慎用自动剔除，建议结合可视化确认。",
             "keywords": ["数值x", "数值y", "维度1", "维度2"], "applicable_fields": ["数值x", "数值y"]},
        ],
        "viz": ["density", "ridge", "scatter", "line", "heatmap", "grouped"],
    },
    {
        "key": "general", "name": "通用（全学科高频基础图表）", "alias": ["通用"],
        "keywords": ["x数值", "y数值", "子组", "数值", "时间", "计数", "维度1", "维度2", "类别", "分组"],
        "directions": [],
        "rules": [
            {"id": "general_basic", "method": "iqr", "factor": 1.5, "grade": "L1",
             "advice": "通用稳健默认；当数据有明确学科归属时，优先选用对应学科专属规则以提升精度。",
             "keywords": ["数值", "x数值", "y数值"], "applicable_fields": ["数值", "x数值", "y数值"]},
        ],
        "viz": ["errorbar", "line", "multiline", "box", "heatmap", "scatter", "pie", "stacked"],
    },
]


def _load_disciplines():
    """学科规则：优先加载同目录 disciplines.json（用户免改码即可增删学科/规则）；
    文件缺失或损坏时回退内嵌副本，保证零依赖仍可运行。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disciplines.json")
    try:
        with io.open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return _DISCIPLINE_REGISTRY_EMBEDDED


DISCIPLINE_REGISTRY = _load_disciplines()



def _discipline_index(token):
    if token is None:
        return None
    t = token.strip().lower()
    if t == "auto":
        return "auto"
    for d in DISCIPLINE_REGISTRY:
        if t == d["key"] or t in [a.lower() for a in d.get("alias", [])] or t == d["name"].lower():
            return d
    for d in DISCIPLINE_REGISTRY:
        if t and t in d["name"].lower():
            return d
    return None


def infer_discipline(columns, sample_rows=None):
    cols = [str(c).lower() for c in (columns or [])]
    best = None
    best_score = -1
    best_hits = []
    for d in DISCIPLINE_REGISTRY:
        hits = [kw for kw in d["keywords"] if any(kw in c for c in cols)]
        sc = len(hits)
        if sc > best_score:
            best_score, best, best_hits = sc, d, hits
    if best_score <= 0:
        best = DISCIPLINE_REGISTRY[-1]
        best_hits = []
    directions = best.get("directions") or []
    best_si = None
    best_shits = []
    if directions:
        bscore = -1
        for j, sub in enumerate(directions):
            hits = [kw for kw in sub["keywords"] if any(kw in c for c in cols)]
            sc = len(hits)
            if sc > bscore:
                bscore, best_si, best_shits = sc, j, hits
        direction_name = directions[best_si]["name"] if best_si is not None else None
    else:
        direction_name = None
    matched = sorted(set(best_hits + best_shits))
    total = len(cols) or 1
    confidence = round(min(1.0, len(matched) / total), 3) if best_score > 0 else 0.0
    reason = ("命中学科关键词：" + "、".join(matched)) if matched else "未命中明确学科关键词，已回退至通用基础规则"
    return {"key": best["key"], "name": best["name"], "direction": direction_name,
            "confidence": confidence, "matched_keywords": matched, "reason": reason}


def _grade_for_rule(method, factor, contamination):
    if method == "iqr":
        return "L3" if (factor or 0) >= 5.0 else ("L2" if (factor or 0) >= 3.0 else "L1")
    if method == "zscore":
        return "L2" if (factor or 0) >= 3.0 else "L1"
    if method == "isolation_forest":
        c = contamination if contamination is not None else 0.05
        return "L3" if c >= 0.10 else ("L2" if c >= 0.05 else "L1")
    return "L2"


def _discipline_rule_for_column(disc, colname):
    c = str(colname).lower()
    best = disc["rules"][0]
    best_score = 0
    for r in disc["rules"]:
        score = sum(1 for kw in (r.get("keywords", []) + r.get("applicable_fields", [])) if kw.lower() in c)
        if score > best_score:
            best_score, best = score, r
    return best


def cmd_discipline(args):
    if getattr(args, "list", False):
        out = []
        for d in DISCIPLINE_REGISTRY:
            out.append({
                "key": d["key"], "name": d["name"], "alias": d.get("alias", []),
                "rules": [{"id": r["id"], "method": r["method"], "factor": r.get("factor"),
                           "contamination": r.get("contamination"), "grade": r.get("grade"),
                           "advice": r.get("advice")} for r in d["rules"]],
                "viz": d["viz"],
            })
        emit({"status": "ok", "task": "discipline",
              "result": {"count": len(out), "disciplines": out}})
        return
    if not args.input:
        die("需指定 --input 进行推断，或 --list 列出全部学科")
    columns, rows = load_rows(args.input)
    inf = infer_discipline(columns, rows[:60])
    disc = _discipline_index(inf["key"])
    emit({"status": "ok", "task": "discipline", "result": {
        "inferred": inf,
        "anomaly_rules": [{"id": r["id"], "method": r["method"], "factor": r.get("factor"),
                           "contamination": r.get("contamination"), "grade": r.get("grade"),
                           "advice": r.get("advice")} for r in disc["rules"]],
        "viz_templates": disc["viz"]}})

def _describe(columns, rows):
    out = OrderedDict()
    for c in columns:
        xs = _col_floats(rows, c, dropna=True)
        if not xs:
            continue
        out[c] = {
            "count": len(xs), "missing": sum(1 for r in rows if to_float(r.get(c)) is None),
            "n_inf": sum(1 for x in xs if math.isinf(x)),
            "mean": round(mean(xs), 4), "std": round(stdev(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4),
            "median": round(median(xs), 4),
            "ci95_mean": _mean_ci(xs),
        }
    return out


def _lcomb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom(k, N, K, n):
    if k < 0 or k > K or n - k < 0 or n - k > N - K:
        return 0.0
    return math.exp(_lcomb(K, k) + _lcomb(N - K, n - k) - _lcomb(N, n))


def _fisher_exact_2x2(m):
    """2×2 Fisher 精确检验（双侧，零依赖）。m = [[a,b],[c,d]]。"""
    a, b = int(m[0][0]), int(m[0][1])
    c, d = int(m[1][0]), int(m[1][1])
    N = a + b + c + d
    if N <= 0:
        return None
    r1, c1 = a + b, a + c
    obs = _hypergeom(a, N, r1, c1)
    if obs <= 0:
        return None
    lo, hi = max(0, c1 - (N - r1)), min(r1, c1)
    p = 0.0
    for x in range(lo, hi + 1):
        px = _hypergeom(x, N, r1, c1)
        if px <= obs:
            p += px
    return min(1.0, p)


def _bf10_bic(n, sse_null, sse_alt, k_null, k_alt):
    """BIC 近似贝叶斯因子 BF10（零依赖，替代二元显著性）。"""
    if sse_null <= 0 or sse_alt <= 0 or n <= 0:
        return None
    bic_null = n * math.log(sse_null / n) + k_null * math.log(n)
    bic_alt = n * math.log(sse_alt / n) + k_alt * math.log(n)
    return round(math.exp((bic_null - bic_alt) / 2.0), 4)


def _remove_l3_outliers(rows, group_col, value_col):
    """按分组对各组数值列用 IQR×3.0（L3）剔除离群点，返回过滤后行。"""
    groups = _group_values(rows, group_col, value_col)
    bounds = {}
    for g, vals in groups.items():
        if len(vals) < 4:
            bounds[g] = (float("-inf"), float("inf"))
            continue
        xs = sorted(vals)
        q1 = quantile(xs, 0.25); q3 = quantile(xs, 0.75)
        iqr = q3 - q1
        bounds[g] = (q1 - 3.0 * iqr, q3 + 3.0 * iqr)
    kept = []
    for r in rows:
        g = str(r.get(group_col)); v = to_float(r.get(value_col))
        if g in bounds and v is not None and bounds[g][0] <= v <= bounds[g][1]:
            kept.append(r)
    return kept


def _rank_array_np(col):
    """对含 NaN 的 1D numpy 数组做平均秩（与 _ranks 语义一致），NaN 处仍返回 NaN。"""
    out = np.full(col.shape, np.nan)
    mask = ~np.isnan(col)
    vals = col[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    sv = vals[order]
    ranks_sorted = np.empty(sv.shape, dtype=float)
    i = 0
    n = sv.size
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based 平均秩
        ranks_sorted[i:j + 1] = avg
        i = j + 1
    res = np.empty(vals.shape, dtype=float)
    res[order] = ranks_sorted
    out[mask] = res
    return out


def _corr_matrices_np(num, series):
    """用 numpy 一次性算出 Pearson/Spearman 相关矩阵（成对完整，NaN 处留 nan）。

    返回 (pear, spear) 两个 (k x k) numpy 数组；任何异常都回退为 (None, None)，
    交由纯 Python 路径兜底。零依赖前提下不强制依赖 numpy。
    """
    try:
        import numpy as np
    except Exception:
        return None, None
    try:
        k = len(num)
        if k == 0:
            return None, None
        cols = []
        for c in num:
            arr = np.array([(x if (isinstance(x, (int, float)) and x == x) else np.nan)
                            for x in series[c]], dtype=float)
            cols.append(arr)
        A = np.column_stack(cols)
        valid = ~np.isnan(A)
        pear = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1, k):
                m = valid[:, i] & valid[:, j]
                if int(m.sum()) < 2:
                    continue
                r = float(np.corrcoef(A[m, i], A[m, j])[0, 1])
                pear[i, j] = r
                pear[j, i] = r
            pear[i, i] = 1.0
        ranks = [_rank_array_np(A[:, i]) for i in range(k)]
        spear = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1, k):
                m = valid[:, i] & valid[:, j]
                if int(m.sum()) < 3:
                    continue
                r = float(np.corrcoef(ranks[i][m], ranks[j][m])[0, 1])
                spear[i, j] = r
                spear[j, i] = r
            spear[i, i] = 1.0
        return pear, spear
    except Exception:
        return None, None


def _corr_matrix(columns, rows, permutation=False, n_perm=2000, seed=0):
    num = _numeric_columns(columns, rows)
    idx = {c: i for i, c in enumerate(num)}  # 查表替 num.index，避免 O(k) 查找拖到 O(k^3)
    pear = OrderedDict()
    spear = OrderedDict()
    p_pear = OrderedDict()
    p_spear = OrderedDict()
    p_adj = OrderedDict()
    p_perm_map = OrderedDict()
    ci_pear = OrderedDict()
    ci_spear = OrderedDict()
    bf10 = OrderedDict()
    series = _numeric_series(rows, num)
    # numpy 加速路径：一次性算出 Pearson/Spearman 矩阵；无 numpy 或异常时回退纯 Python
    _np_pear, _np_spear = _corr_matrices_np(num, series)

    def _sp(a, b):
        pairs = [(x, y) for x, y in zip(series[a], series[b]) if x is not None and y is not None]
        if len(pairs) < 3:
            return float("nan"), None
        rx = _ranks([x for x, _ in pairs])
        ry = _ranks([y for _, y in pairs])
        r = pearson(rx, ry)
        denom = 1 - r * r
        if denom <= 1e-12:
            return r, 0.0
        t = abs(r) * math.sqrt((len(pairs) - 2) / denom)
        return r, _t_two_sided_p(t, len(pairs) - 2)

    def _p_from_r(r, n):
        if n < 3 or r != r:
            return None
        denom = 1 - r * r
        if denom <= 1e-12:
            return 0.0
        t = abs(r) * math.sqrt((n - 2) / denom)
        return _t_two_sided_p(t, n - 2)

    pair_list = []  # (a, b, pearson_p) 唯一对，用于 BH 多重比较校正
    for a in num:
        pear[a] = {}; spear[a] = {}; p_pear[a] = {}; p_spear[a] = {}; p_adj[a] = {}
        p_perm_map[a] = {}; ci_pear[a] = {}; ci_spear[a] = {}; bf10[a] = {}
        for b in num:
            if a == b:
                r_self = (_np_pear[idx[a], idx[a]]
                          if _np_pear is not None else pearson(series[a], series[b]))
                pear[a][b] = 1.0 if r_self == r_self else None
                spear[a][b] = 1.0 if r_self == r_self else None
                p_pear[a][b] = None; p_spear[a][b] = None; p_adj[a][b] = None
                ci_pear[a][b] = None; ci_spear[a][b] = None
                bf10[a][b] = None
                continue
            avail = [(x, y) for x, y in zip(series[a], series[b]) if x is not None and y is not None]
            n = len(avail)
            if _np_pear is not None:
                ia, ib = idx[a], idx[b]
                r_ = float(_np_pear[ia, ib])
                _rs = _np_spear[ia, ib]
                rs = float(_rs) if _rs == _rs else None
                ps_ = _p_from_r(rs, n)
            else:
                r_ = pearson(series[a], series[b])
                rs, ps_ = _sp(a, b)
            pp_ = _p_from_r(r_, n)
            pear[a][b] = round(r_, 3)
            spear[a][b] = round(rs, 3) if (rs is not None and rs == rs) else None
            p_pear[a][b] = _r4(pp_)
            p_spear[a][b] = _r4(ps_)
            ci_pear[a][b] = _fisher_z_ci(r_, n)
            ci_spear[a][b] = _fisher_z_ci(rs, n) if rs is not None else None
            if permutation and n >= 5 and n_perm > 0:
                xsa, ysa = [], []
                for x, y in zip(series[a], series[b]):
                    if x is not None and y is not None:
                        xsa.append(x); ysa.append(y)
                rng = random.Random(seed)
                cnt = 0
                for pi in range(n_perm):
                    if pi % 500 == 0 and pi > 0:
                        _vlog("  permutation (corr): %d/%d" % (pi, n_perm))
                    perm = ysa[:]
                    rng.shuffle(perm)
                    if abs(pearson(xsa, perm)) >= abs(r_):
                        cnt += 1
                pperm = (cnt + 1) / (n_perm + 1)
            else:
                pperm = None
            p_perm_map[a][b] = _r4(pperm)
            # 贝叶斯因子 BF10（BIC 近似）：相关视作 Y~X 简单回归，复用已验证的 _bf10_bic
            ys = [x for x in series[b] if x is not None]
            m_ys = mean(ys)
            sst = sum((x - m_ys) ** 2 for x in ys) if ys else 0.0
            sse_alt = sst * (1 - r_ * r_) if sst > 0 else 0.0
            bf = _bf10_bic(n, sst, sse_alt, 1, 2) if sst > 0 else None
            bf10[a][b] = _r4(bf)
            if idx[a] < idx[b] and pp_ is not None:
                pair_list.append((a, b, pp_))
    # Benjamini-Hochberg (FDR) 校正，控制多次成对检验的假发现率
    m = len(pair_list)
    adj_map = {}
    if m > 0:
        pvals = [p for _, _, p in pair_list]
        order = sorted(range(m), key=lambda i: pvals[i])
        ranked = [0.0] * m
        prev = 1.0
        for j in range(m - 1, -1, -1):
            i = order[j]
            prev = min(prev, pvals[i] * m / (j + 1))
            ranked[i] = prev
        for i in range(m):
            a, b, _ = pair_list[i]
            adj_map[(a, b)] = _r4(min(1.0, ranked[i]))
        for (a, b), v in adj_map.items():
            p_adj[a][b] = v
            p_adj[b][a] = v
    # 强相关判定（结合 FDR 校正后显著性）
    strong = []
    for a in num:
        for b in num[idx[a] + 1:]:
            r_ = pear[a][b]
            v = p_adj[a][b]
            if r_ == r_ and abs(r_) > 0.5:
                if v is not None:
                    tag = "（FDR 校正 p=%.4f，%s）" % (v, "显著" if v < _CFG["alpha"] else "不显著")
                else:
                    tag = ""
                strong.append("%s~%s(r=%.3f)%s" % (a, b, r_, tag))
    if strong:
        interp = "Pearson 强相关对（|r|>0.5）：" + "、".join(strong)
    else:
        interp = "各变量间未见 |r|>0.5 的强线性相关（已对全部成对检验做 FDR/BH 校正）"
    interp += "；若变量非正态或为有序变量，建议参考 spearman 矩阵；矩阵同时给出每对 p 值与 BH 校正后 p_adj（adjust_method）"
    return {"pearson": pear, "spearman": spear,
            "p_pearson": p_pear, "p_spearman": p_spear, "p_adj_fdr": p_adj,
            "p_permutation": p_perm_map,
            "bayes_factor_10": bf10,
            "ci_pearson": ci_pear, "ci_spearman": ci_spear,
            "adjust_method": "benjamini_hochberg",
            "interpretation": interp}


def _normality(columns, rows):
    out = OrderedDict()
    for c in columns:
        xs = _col_floats(rows, c, dropna=True)
        if len(xs) < 3:
            continue
        s = stdev(xs)
        if s == 0:
            out[c] = {"note": "方差为0", "normal": None}
            continue
        r = _normality_one(xs, with_kurtosis=True)
        out[c] = r
    non_normal = [c for c, v in out.items() if v.get("normal") is False]
    out["interpretation"] = (
        ("非正态列：" + "、".join(non_normal) + "；建议参数检验前先变换（skew 子命令）或改用非参数方法/Spearman 相关")
        if non_normal else "各数值列均未拒绝正态性假设（α=%g）" % _CFG["alpha"])
    return out


def _levene(groups, center="mean"):
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return None
    res = _scipy_try(lambda: spstats.levene(*groups, center=center))
    if res is not None:
        w, p = res
        return {"W": round(float(w), 4), "p_value": round(float(p), 4),
                "df1": k - 1, "df2": N - k, "equal_var": bool(p > _CFG["alpha"])}
    centers = [mean(g) for g in groups] if center == "mean" else [median(g) for g in groups]
    z = [[abs(x - centers[i]) for x in groups[i]] for i in range(k)]
    zbar = [mean(zi) for zi in z]
    ztot = mean([x for zi in z for x in zi])
    ssb = sum(len(g) * (zb - ztot) ** 2 for g, zb in zip(groups, zbar))
    ssw = sum((x - zbar[i]) ** 2 for i, zi in enumerate(z) for x in zi)
    df1, df2 = k - 1, N - k
    F = (ssb / df1) / (ssw / df2) if ssw > 0 else float("inf")
    p = _f_sf(F, df1, df2)
    return {"W": round(F, 4), "p_value": (_r4(p)),
            "df1": df1, "df2": df2, "equal_var": bool(p is not None and p > _CFG["alpha"])}


def _ks_d_stat(xs, m, s):
    """单样本 KS 型 D 统计量：经验 CDF 与 N(m,s) 理论 CDF 的最大竖直距离。"""
    n = len(xs)
    if n == 0 or not s or s != s:
        return None
    z = sorted((x - m) / s for x in xs)
    D = 0.0
    for i in range(n):
        cdf = _norm_cdf(z[i])
        d1 = abs((i + 1) / n - cdf)
        d2 = abs(i / n - cdf)
        if d1 > D:
            D = d1
        if d2 > D:
            D = d2
    return D


def _anderson_darling(xs):
    """Anderson-Darling 正态性检验（零依赖）：返回 (A2, A2_star, p_value)；样本不足返回 None。
    p 用 Stephens(1974) 对未知均值/方差修正统计量的近似公式。"""
    n = len(xs)
    if n < 3:
        return None
    m = mean(xs); s = stdev(xs)
    if not s or s != s:
        return None
    z = sorted((x - m) / s for x in xs)
    s_term = 0.0
    for i in range(n):
        lo = _norm_cdf(z[i])
        hi = _norm_cdf(z[n - 1 - i])
        lo = max(1e-12, min(1.0 - 1e-12, lo))
        hi = max(1e-12, min(1.0 - 1e-12, hi))
        s_term += (2 * (i + 1) - 1) * (math.log(lo) + math.log(1.0 - hi))
    A2 = -n - s_term / n
    if A2 < 0:
        A2 = 0.0
    A2s = A2 * (1.0 + 4.0 / n - 25.0 / (n * n))
    if A2s >= 0.6:
        p = math.exp(1.2937 - 5.709 * A2s + 0.0186 * A2s * A2s)
    elif A2s >= 0.34:
        p = math.exp(0.9177 - 4.279 * A2s - 1.38 * A2s * A2s)
    elif A2s >= 0.2:
        p = 1.0 - math.exp(-8.318 + 42.796 * A2s - 59.938 * A2s * A2s)
    else:
        p = 1.0 - math.exp(-13.436 + 101.14 * A2s - 223.73 * A2s * A2s)
    p = max(0.0, min(1.0, p))
    return (A2, A2s, p)


def _lilliefors(xs, seed=0, b=None):
    """Lilliefors 正态性检验（KS 但均值/标准差由样本估计）。返回 (D, p_bootstrap)；
    p 用参数自助法（从 N(m,s) 重抽 B 次，比例 ≥ 观测 D 即 p）。零依赖、种子固定可复现。"""
    n = len(xs)
    if n < 4:
        return None
    m = mean(xs); s = stdev(xs)
    if not s or s != s:
        return None
    D = _ks_d_stat(xs, m, s)
    if D is None:
        return None
    if b is None:
        b = max(200, min(2000, 500000 // n))
    rng = random.Random(seed)
    ge = 0
    for _ in range(b):
        samp = [m + s * _norm_ppf(rng.random()) for _ in range(n)]
        ms = mean(samp); ss = stdev(samp)
        if not ss:
            ge += 1
            continue
        Ds = _ks_d_stat(samp, ms, ss)
        if Ds is not None and Ds >= D:
            ge += 1
    return (D, ge / b if b > 0 else None)


def _normality_one(xs, with_kurtosis=False):
    xs = [x for x in xs if x is not None and x == x]
    if len(xs) < _CFG["min_normality_n"]:
        return {"normal": None, "note": "样本不足"}
    m, s = mean(xs), stdev(xs)
    if not s or s != s:
        return {"normal": None, "note": "方差为0"}
    sk = sum((x - m) ** 3 for x in xs) / (len(xs) * s ** 3)
    out = OrderedDict()
    if with_kurtosis:
        out["skewness"] = round(sk, 4)
        out["kurtosis"] = round(sum((x - m) ** 4 for x in xs) / (len(xs) * s ** 4) - 3, 4)
    # Anderson-Darling（对尾部最敏感，零依赖解析 p 值）
    ad = _anderson_darling(xs)
    if ad is not None:
        A2, A2s, adp = ad
        out["anderson_darling"] = {"A2": round(A2, 4), "A2_star": round(A2s, 4),
                                  "p": _r4(adp), "normal": bool(adp > _CFG["alpha"])}
    # Lilliefors（KS 但均值/标准差由样本估计，参数自助 p 值）
    lf = _lilliefors(xs)
    if lf is not None:
        Dl, lfp = lf
        out["lilliefors"] = {"D": round(Dl, 4), "p_bootstrap": _r4(lfp),
                             "normal": bool(lfp is not None and lfp > _CFG["alpha"])}
    if HAS_SCIPY:
        try:
            cap = _CFG["shapiro_max_n"]
            w, p = spstats.shapiro(xs[:cap])
            out.update({"shapiro_W": round(float(w), 4), "shapiro_p": round(float(p), 4),
                        "normal": bool(p > _CFG["alpha"]), "method": "shapiro"})
            if len(xs) > cap:
                out["truncated_to"] = cap
            return out
        except Exception:
            pass
    out.setdefault("skewness", round(sk, 4))
    ku = sum((x - m) ** 4 for x in xs) / (len(xs) * s ** 4) - 3
    out.setdefault("kurtosis", round(ku, 4))
    # 零依赖回退：Jarque-Bera 近似（同时校验偏度与超额峰度），比单纯偏度阈值更稳健
    n_ = len(xs)
    jb = n_ / 6.0 * (sk ** 2 + ku ** 2 / 4.0) if n_ > 0 else 0.0
    jbp = _chi2_sf(jb, 2)
    normal = bool(jbp is not None and jbp == jbp and jbp > _CFG["alpha"]
                  and abs(sk) < _CFG["skew_rule_cut"])
    out.update({"jarque_bera": round(jb, 4), "jarque_bera_p": _r4(jbp),
                "normal": normal, "method": "jarque_bera"})
    return out


def _anova(columns, rows, spec, ss_type="III"):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 2:
        return {"error": "anova 需 '因子,数值' 或 '因子A,因子B,数值'"}
    if len(parts) == 3:
        fa, fb, fv = parts
        a = [str(r.get(fa)) for r in rows]
        b = [str(r.get(fb)) for r in rows]
        y = [to_float(r.get(fv)) for r in rows]
        triples = [(ai, bi, yi) for ai, bi, yi in zip(a, b, y) if yi is not None]
        return _two_way_anova(triples, ss_type)
    else:
        fa, fv = parts[0], parts[-1]
        a = [str(r.get(fa)) for r in rows]
        y = [to_float(r.get(fv)) for r in rows]
        pairs = [(ai, yi) for ai, yi in zip(a, y) if yi is not None]
        return _one_way_anova(pairs)


def _one_way_anova(pairs):
    groups = OrderedDict()
    for ai, yi in pairs:
        groups.setdefault(ai, []).append(yi)
    k = len(groups)
    n = len(pairs)
    if k < 2 or n <= k:
        return {"error": "分组数或样本不足"}
    min_group_n = min(len(g) for g in groups.values())
    grand = mean([y for _, y in pairs])
    sst = sum((y - grand) ** 2 for _, y in pairs)
    ssa = sum(len(g) * (mean(g) - grand) ** 2 for g in groups.values())
    sse = sst - ssa
    dfa, dfe = k - 1, n - k
    msa, mse = ssa / dfa, sse / dfe
    perfect_fit = bool(mse == 0)
    F = msa / mse if mse > 0 else float("inf")
    p = _f_sf(F, dfa, dfe)
    sig = _sig(p)

    norm = OrderedDict()
    all_normal = True
    for lv, g in groups.items():
        nr = _normality_one(g)
        norm[lv] = nr
        if nr.get("normal") is not True:
            all_normal = False

    lev = _levene(list(groups.values()))
    equal_var = lev.get("equal_var", True) if lev else True

    if all_normal and equal_var:
        recommend, reason = "anova", "各组满足正态性且方差齐，适用标准单因素 ANOVA"
    elif equal_var and not all_normal:
        recommend, reason = "kruskal", "方差齐但存在非正态组，建议非参数 Kruskal-Wallis"
    else:
        recommend, reason = "welch", "方差不齐，建议 Welch ANOVA（或 Kruskal-Wallis）"

    ws = []
    w_tot = 0.0
    for g in groups.values():
        ni = len(g)
        vi = stdev(g) ** 2 if len(g) > 1 else 0.0
        if vi <= 0:
            vi = 1e-9
        wi = ni / vi
        ws.append((ni, mean(g), vi, wi))
        w_tot += wi
    xw = sum(wi * mi for ni, mi, vi, wi in ws) / w_tot
    num = sum(wi * (mi - xw) ** 2 for ni, mi, vi, wi in ws)
    den = 1.0 + (2.0 * (k - 1) / (k * k - 1)) * sum(
        ((1 - wi / w_tot) ** 2 / (ni - 1 if ni > 1 else 1.0)) for ni, mi, vi, wi in ws)
    W = (num / (k - 1)) / den if den > 0 else float("inf")
    df1w = k - 1
    df2w = (k * k - 1) / (3 * den) if den > 0 else float("inf")
    pw = _f_sf(W, df1w, df2w)

    flat = [y for _, y in pairs]
    rk = _ranks(flat)

    grp_ranks = {}
    for (lv, _), r_ in zip(pairs, rk):
        grp_ranks.setdefault(lv, []).append(r_)
    Rbar = {lv: mean(rs) for lv, rs in grp_ranks.items()}
    Ntot = n
    H = (12.0 / (Ntot * (Ntot + 1))) * sum(len(g) * Rbar[lv] ** 2 for lv, g in groups.items()) - 3 * (Ntot + 1)
    tie_corr = sum(c ** 3 - c for c in Counter(flat).values() if c > 1)
    if tie_corr:
        H = H / (1 - tie_corr / (Ntot ** 3 - Ntot)) if (Ntot ** 3 - Ntot) else H
    pk = _chi2_sf(H, k - 1)

    eta2 = ssa / sst if sst > 0 else None
    omega2 = (ssa - dfa * mse) / (sst + mse) if (sst + mse) > 0 else None

    final_p, final_name = {
        "anova": (p, "单因素 ANOVA"),
        "welch": (pw, "Welch ANOVA"),
        "kruskal": (pk, "Kruskal-Wallis 检验"),
    }[recommend]
    bf10 = _bf10_bic(n, sst, sse, 1, k)
    if recommend != "anova":
        gate_dev = ("若强行采用标准 ANOVA（已不满足其前提），p=%.4f；与推荐 %s(p=%.4f) 的显著性结论%s"
                    % (p, final_name, final_p,
                       "一致" if _sig(p) == _sig(final_p) else "不一致（建议以推荐方法为准）"))
    else:
        gate_dev = None
    interp = "%s；按门控采用%s，p=%s，组间差异%s（α=%g）" % (
        reason, final_name, _fmt_p(final_p), "显著" if _sig(final_p) else "不显著",
        _CFG["alpha"])
    if eta2 is not None:
        interp += "；效应量 η²=%.3f（%s效应）" % (eta2, _effect_label("eta2", eta2))
    if omega2 is not None:
        interp += "；ω²=%.3f（更无偏的效应量）" % omega2
    if _sig(final_p) and k > 2:
        interp += "。建议进行多重比较（--tukey）定位差异来源"
    if gate_dev:
        interp += "；" + gate_dev
    php = _posthoc_power_anova(eta2, n, k)
    reasons = []
    if min_group_n < 5:
        reasons.append("存在样本量<5的分组（最少 %d 条），统计功效可能不足" % min_group_n)
    if php is not None and php < 0.5:
        reasons.append("事后统计功效偏低（=%.2f < 0.5），当前观测效应易被漏检" % php)
    return {
        "type": "one_way", "factor_levels": k, "n": n,
        "perfect_fit": perfect_fit,
        "posthoc_power": php,
        "bayes_factor_10": bf10,
        "gate_deviation": gate_dev,
        "low_evidence_advisory": _low_evidence_advisory_text(reasons),
        "SSA": round(ssa, 4), "SSE": round(sse, 4), "SST": round(sst, 4),
        "F": round(F, 4), "p_value": (_r4(p)), "significant": sig,
        "eta_squared": _r4(eta2),
        "omega_squared": _r4(omega2),
        "effect_size_label": _effect_label("eta2", eta2),
        "normality": norm, "all_normal": all_normal,
        "levene": lev,
        "gate": {"recommend": recommend, "reason": reason,
                 "welch_significant": _sig(pw),
                 "kruskal_significant": _sig(pk)},
        "welch": {"F": round(W, 4), "df1": round(df1w, 2), "df2": round(df2w, 2),
                  "p_value": (_r4(pw)),
                  "significant": _sig(pw)},
        "kruskal": {"H": round(H, 4), "df": k - 1,
                    "p_value": (_r4(pk)),
                    "significant": _sig(pk),
                    "epsilon_squared": _r4((H - k + 1) / (n - k)) if (n - k) > 0 else None,
                    "epsilon_squared_label": _effect_label("eta2", (H - k + 1) / (n - k)) if (n - k) > 0 else None,
                    "posthoc": (_kw_posthoc(groups) if k > 2 else None)},
        "interpretation": interp,
    }


def _two_way_anova(triples, ss_type="III"):
    a_levels = sorted(set(a for a, _, _ in triples))
    b_levels = sorted(set(b for _, b, _ in triples))
    yv = [y for _, _, y in triples]
    n = len(triples)
    A, B = len(a_levels), len(b_levels)
    a_idx = {lv: i for i, lv in enumerate(a_levels)}
    b_idx = {lv: i for i, lv in enumerate(b_levels)}
    if A < 2 or B < 2:
        return {"error": "双因素 ANOVA 需每个因子至少 2 个水平（当前 A=%d, B=%d）" % (A, B),
                "type": "two_way", "ss_type": ss_type}

    def _effects_design(drop=()):
        X = []
        for a, b, _ in triples:
            ia, ib = a_idx[a], b_idx[b]
            row = [1.0]
            if "A" not in drop:
                for j in range(A - 1):
                    row.append(1.0 if ia == j else (-1.0 if ia == A - 1 else 0.0))
            if "B" not in drop:
                for j in range(B - 1):
                    row.append(1.0 if ib == j else (-1.0 if ib == B - 1 else 0.0))
            if "AB" not in drop:
                for ja in range(A - 1):
                    for jb in range(B - 1):
                        va = 1.0 if ia == ja else (-1.0 if ia == A - 1 else 0.0)
                        vb = 1.0 if ib == jb else (-1.0 if ib == B - 1 else 0.0)
                        row.append(va * vb)
            X.append(row)
        return X

    if ss_type == "I":

        def build_X(terms):
            X = []
            for a, b, _ in triples:
                row = [1.0]
                if "A" in terms:
                    for lv in a_levels:
                        if lv != a_levels[0]:
                            row.append(1.0 if a == lv else 0.0)
                if "B" in terms:
                    for lv in b_levels:
                        if lv != b_levels[0]:
                            row.append(1.0 if b == lv else 0.0)
                if "AB" in terms:
                    for la in a_levels:
                        if la != a_levels[0]:
                            for lb in b_levels:
                                if lb != b_levels[0]:
                                    row.append(1.0 if (a == la and b == lb) else 0.0)
                X.append(row)
            return X
        def sse_of(terms):
            return _ols_sse(build_X(terms), yv)
        sse0 = sse_of([])
        sseA = sse_of(["A"])
        sseAB = sse_of(["A", "B"])
        sseABi = sse_of(["A", "B", "AB"])
        ssA, ssB, ssAB = sse0 - sseA, sseA - sseAB, sseAB - sseABi
        dfA, dfB, dfAB = A - 1, B - 1, (A - 1) * (B - 1)
        dfE = n - (1 + dfA + dfB + dfAB)
        mse = sseABi / dfE if dfE > 0 else float("inf")
        FA = ssA / dfA / mse if mse > 0 else float("inf")
        FB = ssB / dfB / mse if mse > 0 else float("inf")
        FAB = ssAB / dfAB / mse if (mse > 0 and dfAB > 0) else float("inf")
    else:

        sse_full = _ols_sse(_effects_design(), yv)
        sse_noA = _ols_sse(_effects_design(drop=("A",)), yv)
        sse_noB = _ols_sse(_effects_design(drop=("B",)), yv)
        sse_noAB = _ols_sse(_effects_design(drop=("AB",)), yv)
        if sse_full == float("inf"):
            return {"error": "设计非满秩（存在空单元格或因子组合不完整），III 型效应不可估计；请改用 --ss-type I 或补全数据",
                    "type": "two_way", "ss_type": "III"}
        ssA, ssB, ssAB = sse_noA - sse_full, sse_noB - sse_full, sse_noAB - sse_full
        dfA, dfB, dfAB = A - 1, B - 1, (A - 1) * (B - 1)
        dfE = n - A * B
        if dfE <= 0:
            return {"error": "无残差自由度（每单元格仅 1 条观测），无法检验；III 型需重复观测",
                    "type": "two_way", "ss_type": "III"}
        mse = sse_full / dfE
        FA = ssA / dfA / mse if mse > 0 else float("inf")
        FB = ssB / dfB / mse if mse > 0 else float("inf")
        FAB = ssAB / dfAB / mse if (mse > 0 and dfAB > 0) else float("inf")
    pA = _f_sf(FA, dfA, dfE)
    pB = _f_sf(FB, dfB, dfE)
    pAB = _f_sf(FAB, dfAB, dfE)
    sse_res = sse_full if ss_type != "I" else sseABi

    def _peta(ss):
        return ss / (ss + sse_res) if (ss + sse_res) > 0 else None
    petaA, petaB, petaAB = _peta(ssA), _peta(ssB), _peta(ssAB)
    seg = []
    for nm, pv, pe in (("A 主效应", pA, petaA), ("B 主效应", pB, petaB), ("A×B 交互", pAB, petaAB)):
        s = "%s p=%s（%s" % (nm, _fmt_p(pv), "显著" if _sig(pv) else "不显著")
        if pe is not None:
            s += "，偏η²=%.3f" % pe
        seg.append(s + "）")
    interp = "双因素 ANOVA（%s 型平方和）：%s" % (ss_type, "；".join(seg))
    if _sig(pAB):
        interp += "。交互效应显著，主效应解释需谨慎，建议做简单效应分析"
    return {"type": "two_way", "ss_type": ss_type, "a_levels": A, "b_levels": B,
            "SSA": round(ssA, 4), "SSB": round(ssB, 4), "SSAB": round(ssAB, 4),
            "SSE": round(sse_res, 4),
            "FA": round(FA, 4), "FB": round(FB, 4), "FAB": round(FAB, 4),
            "p_A": _r4(pA),
            "p_B": _r4(pB),
            "p_AB": _r4(pAB),
            "significant_A": _sig(pA),
            "significant_B": _sig(pB),
            "significant_AB": _sig(pAB),
            "partial_eta_sq_A": _r4(petaA),
            "partial_eta_sq_B": _r4(petaB),
            "partial_eta_sq_AB": _r4(petaAB),
            "interpretation": interp}


def _ols_sse(X, y):
    n = len(X)
    p = len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(p)]
    beta = gaussian_solve(XtX, Xty)
    if beta is None:
        return float("inf")
    pred = [sum(beta[j] * X[r][j] for j in range(p)) for r in range(n)]
    return sum((y[r] - pred[r]) ** 2 for r in range(n))

_QT = {
    2: [3.64, 3.15, 3.01, 2.95, 2.89, 2.83, 2.77],
    3: [4.60, 3.88, 3.67, 3.58, 3.49, 3.40, 3.31],
    4: [5.22, 4.33, 4.08, 3.96, 3.85, 3.74, 3.63],
    5: [5.67, 4.65, 4.37, 4.24, 4.11, 3.98, 3.86],
    6: [6.03, 4.91, 4.60, 4.46, 4.32, 4.18, 4.05],
    7: [6.33, 5.12, 4.79, 4.64, 4.49, 4.34, 4.21],
    8: [6.58, 5.30, 4.95, 4.79, 4.64, 4.48, 4.33],
    9: [6.80, 5.47, 5.09, 4.92, 4.77, 4.60, 4.45],
    10: [7.00, 5.60, 5.22, 5.04, 4.88, 4.71, 4.56],
}
_QDF = [5, 10, 15, 20, 30, 60, 10 ** 9]


def _q_table(k, df):
    if k not in _QT or df is None or df <= 0:
        return None
    row = _QT[k]
    if df <= _QDF[0]:
        return row[0]
    if df >= _QDF[-1]:
        return row[-1]
    for i in range(len(_QDF) - 1):
        if _QDF[i] <= df <= _QDF[i + 1]:
            t = (df - _QDF[i]) / (_QDF[i + 1] - _QDF[i])
            return row[i] + (row[i + 1] - row[i]) * t
    return row[-1]


def _tukey(columns, rows, spec, alpha=None):
    alpha = _CFG["alpha"] if alpha is None else alpha
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "tukey 需 factor,value"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) < 2:
        return {"error": "分组不足"}
    names = list(groups.keys())
    k = len(names)

    lev = _levene(list(groups.values()))
    equal_var = lev.get("equal_var", True) if lev else True
    stat = {nm: (mean(g), (stdev(g) ** 2 if len(g) > 1 else 1e-9), len(g)) for nm, g in groups.items()}
    ssw = sum((x - stat[nm][0]) ** 2 for nm, g in groups.items() for x in g)
    dfw = sum(len(g) for g in groups.values()) - k
    mse = ssw / dfw if dfw > 0 else float("inf")
    use_gh = not equal_var
    method_name = "games_howell" if use_gh else "tukey_hsd"
    pairs = []
    SQ2 = math.sqrt(2.0)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = names[i], names[j]
            ma, sa2, na = stat[a]
            mb, sb2, nb = stat[b]
            diff = ma - mb
            if not use_gh:

                se_q = math.sqrt(mse / 2.0 * (1.0 / na + 1.0 / nb))
                q_obs = abs(diff) / se_q if se_q > 0 else float("inf")
                dfq = dfw
            else:

                se_raw = math.sqrt(sa2 / na + sb2 / nb)
                se_q = se_raw / SQ2
                q_obs = abs(diff) / se_q if se_q > 0 else float("inf")
                dfq = (sa2 / na + sb2 / nb) ** 2 / (
                    (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)) if (na > 1 and nb > 1) else float("inf")
            if HAS_SR and dfq != float("inf"):
                q_crit = float(studentized_range.ppf(1 - alpha, k, dfq))
                p = float(studentized_range.sf(q_obs, k, dfq))
            else:

                q_crit = (_q_table(k, dfq) if dfq != float("inf") else None) or (3.0 if k <= 5 else 3.5)

                p = _t_two_sided_p(q_obs / SQ2, dfq) if dfq != float("inf") else None
            sig = bool(q_obs > q_crit)

            ci = ([round(diff - q_crit * se_q, 4), round(diff + q_crit * se_q, 4)]
                  if (se_q > 0 and q_crit is not None) else None)
            item = {"group_a": a, "group_b": b, "mean_diff": round(diff, 4),
                    "q_observed": round(q_obs, 4) if q_obs != float("inf") else None,
                    "q_critical": _r4(q_crit),
                    "p_value": (_r4(p)),
                    "significant": sig,
                    "ci95_mean_diff": ci}
            if not use_gh:
                item["HSD_threshold"] = round(q_crit * se_q, 4) if se_q > 0 else None
            else:
                item["df"] = round(dfq, 2) if dfq != float("inf") else None
            pairs.append(item)
    sig_pairs = ["%s vs %s(p=%s)" % (x["group_a"], x["group_b"], _fmt_p(x["p_value"]))
                 for x in pairs if x["significant"]]
    interp = "%s 多重比较（Levene p=%s，%s）：%s" % (
        "Tukey HSD" if not use_gh else "Games-Howell",
        _fmt_p(lev.get("p_value") if lev else None),
        "方差齐" if equal_var else "方差不齐",
        ("差异显著的组对：" + "、".join(sig_pairs)) if sig_pairs
        else "各组对差异均不显著（α=%g）" % alpha)
    note = ("p 值为精确值（scipy studentized range）" if HAS_SR else
            "零依赖回退：临界值取近似表插值，p 值为未校正 t 近似（仅供参考）；安装 scipy 可得精确 p")
    if not HAS_SR and abs(alpha - 0.05) > 1e-12:
        note += "；零依赖 q 临界值表仅支持 α=0.05，当前 α=%g 下临界值仍按 0.05 取值" % alpha
    return {"method": method_name, "equal_var": equal_var, "alpha": alpha,
            "mse": round(mse, 4), "levene_p": (lev.get("p_value") if lev else None),
            "note": note,
            "pairs": pairs,
            "interpretation": interp}


def _ttest_summary(t, df, p, d, ci, extra=None):
    out = {"t": _r4(t), "df": (round(df, 2) if df is not None and df == df else None),
           "p_value": _r4(p), "significant": _sig(p),
           "cohen_d": _r4(d), "effect_size_label": _effect_label("d", d),
           "ci95_mean_diff": ci}
    if extra:
        out.update(extra)
    return out


def _diff_ci(diff, se, df, level=0.95):
    if se is None or se <= 0 or df is None or df != df or df <= 0:
        return None
    tc = _t_ppf(0.5 + level / 2.0, df)
    if tc is None:
        return None
    return [round(diff - tc * se, 4), round(diff + tc * se, 4)]


def _ttest_one_sample(rows, spec):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "ttest-one 需 '字段,参考值'"}
    fld, mu_s = parts
    mu = to_float(mu_s)
    if mu is None:
        return {"error": "参考值必须是数值: %s" % mu_s}
    xs = _col_floats(rows, fld, dropna=True)
    n = len(xs)
    if n < 2:
        return {"error": "样本量不足（<2）"}
    m, s = mean(xs), stdev(xs)
    if not s or s != s:
        return {"error": "标准差为 0，无法检验"}
    se = s / math.sqrt(n)
    t = (m - mu) / se
    df = n - 1
    p = _t_two_sided_p(abs(t), df)
    d = (m - mu) / s
    ci = _diff_ci(m - mu, se, df)
    interp = "单样本 t 检验：样本均值 %.4f vs 参考值 %s，t=%.3f，p=%s，%s（α=%g）；Cohen's d=%.3f（%s效应）" % (
        m, mu_s, t, _fmt_p(p), "差异显著" if _sig(p) else "差异不显著", _CFG["alpha"],
        d, _effect_label("d", d))
    return _ttest_summary(t, df, p, d, ci, {
        "type": "one_sample", "field": fld, "mu": mu, "n": n,
        "mean": round(m, 4), "std": round(s, 4),
        "interpretation": interp})


def _two_sample_se(g1, g2):
    """独立两样本共用的方差齐性判断、合并 / Welch 标准误与自由度。
    返回 dict：n1,n2,m1,m2,v1,v2,lev,equal_var,diff,se,df,method,se_student,df_student。"""
    n1, n2 = len(g1), len(g2)
    m1, m2 = mean(g1), mean(g2)
    v1, v2 = stdev(g1) ** 2, stdev(g2) ** 2
    diff = m1 - m2
    lev = _levene([g1, g2])
    equal_var = lev.get("equal_var", True) if lev else True
    df_student = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df_student if df_student > 0 else 0.0
    se_student = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2)) if sp2 > 0 else 0.0
    if equal_var:
        se, df, method = se_student, df_student, "student"
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        method = "welch"
    return {"n1": n1, "n2": n2, "m1": m1, "m2": m2, "v1": v1, "v2": v2,
            "lev": lev, "equal_var": equal_var, "diff": diff,
            "se": se, "df": df, "method": method,
            "se_student": se_student, "df_student": df_student}


def _mann_whitney(rows, spec, alpha=None):
    """两独立样本 Mann-Whitney U 检验（零依赖非参）+ Cliff's delta / rank-biserial 效应量。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "mannwhitney 需 '因子,数值'"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) != 2:
        return {"error": "Mann-Whitney 需恰好 2 组，当前 %d 组（≥3 组请用 --anova 走 Kruskal-Wallis）" % len(groups)}
    (na, g1), (nb, g2) = groups.items()
    n1, n2 = len(g1), len(g2)
    if n1 < 1 or n2 < 1:
        return {"error": "每组样本量需 ≥1"}
    u1, p = _mann_whitney_u(g1, g2)
    if u1 is None:
        return {"error": "无法计算 Mann-Whitney U（样本不足或存在退化）"}
    delta, rb = _nonparam_effect(g1, g2, u1=u1)
    a = alpha or _CFG["alpha"]
    med1, med2 = median(g1), median(g2)
    interp = ("Mann-Whitney U 检验（两独立样本，零依赖正态近似+结校正）：%s(n=%d, 中位数=%.4f) vs %s(n=%d, 中位数=%.4f)，"
              "U=%.1f，p=%s，%s（α=%g）；Cliff's δ=%.3f（%s效应），rank-biserial=%.3f"
              % (na, n1, med1, nb, n2, med2, u1, _fmt_p(p),
                 "差异显著" if _sig(p, a) else "差异不显著", a,
                 (delta if delta is not None else float('nan')),
                 (_cliff_label(delta) or "NA"),
                 (rb if rb is not None else float('nan'))))
    return {"type": "mann_whitney_u", "factor": fa, "value": fv,
            "group_a": {"name": na, "n": n1, "median": round(med1, 4)},
            "group_b": {"name": nb, "n": n2, "median": round(med2, 4)},
            "U": round(u1, 1), "p_value": _r4(p),
            "cliff_delta": _r4(delta), "rank_biserial": _r4(rb),
            "effect_size_label": _cliff_label(delta),
            "significant": bool(p is not None and _sig(p, a)),
            "interpretation": interp}


def _ttest_independent(rows, spec, permutation=False, n_perm=2000, seed=0):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "ttest 需 '因子,数值'"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) != 2:
        return {"error": "两独立样本 t 检验需恰好 2 组，当前 %d 组（≥3 组请用 --anova）" % len(groups)}
    (na_name, g1), (nb_name, g2) = groups.items()
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "每组样本量需 ≥2"}
    s = _two_sample_se(g1, g2)
    n1, n2, m1, m2 = s["n1"], s["n2"], s["m1"], s["m2"]
    v1, v2, lev = s["v1"], s["v2"], s["lev"]
    equal_var, diff = s["equal_var"], s["diff"]
    se, df, method = s["se"], s["df"], s["method"]
    se_student, df_student = s["se_student"], s["df_student"]
    if se <= 0:
        return {"error": "标准误为 0，无法检验"}
    t = diff / se
    p = _t_two_sided_p(abs(t), df)
    # 贝叶斯因子（BIC 近似，替代二元显著性）与门控量化偏差
    grand = (n1 * m1 + n2 * m2) / (n1 + n2)
    sse_alt = (n1 - 1) * v1 + (n2 - 1) * v2
    sse_null = sse_alt + n1 * (m1 - grand) ** 2 + n2 * (m2 - grand) ** 2
    bf10 = _bf10_bic(n1 + n2, sse_null, sse_alt, 1, 2)
    if method == "welch":
        gate_dev = ("方差不齐，若忽略而强行 Student t：SE=%.4f、df=%d（vs Welch SE=%.4f、df=%.1f），"
                    "方差不齐下 Student t 的显著性可能被%s" % (
                        se_student, df_student, se, df,
                        "高估" if se_student < se else "低估"))
    else:
        gate_dev = None
    if permutation and (n1 + n2) >= 5 and n_perm > 0:
        rng = random.Random(seed)
        pooled = g1 + g2
        cnt = 0
        for pi in range(n_perm):
            if pi % 500 == 0 and pi > 0:
                _vlog("  permutation (t-test): %d/%d" % (pi, n_perm))
            rng.shuffle(pooled)
            a = pooled[:n1]; b = pooled[n1:]
            if abs(mean(a) - mean(b)) >= abs(diff):
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)
    else:
        perm_p = None
    if equal_var:
        pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = diff / pooled_sd if pooled_sd > 0 else None
        effect_name = "Cohen's d"
    else:
        # 方差不齐：改用 Glass's delta（以方差较大组 SD 为分母，更保守），并附 Hedges' g
        s_ref = math.sqrt(v2) if v2 >= v1 else math.sqrt(v1)
        d = diff / s_ref if s_ref > 0 else None
        effect_name = "Glass's delta"
    # Hedges' g（小样本无偏校正值，所有情形给出）
    pooled_sd_all = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    g = (diff / pooled_sd_all * (1 - 3.0 / (4 * (n1 + n2) - 9))) if (pooled_sd_all > 0 and (n1 + n2) > 2) else None
    ci = _diff_ci(diff, se, df)
    summary_d = d if equal_var else None
    interp = "两独立样本 t 检验（Levene p=%s，%s→采用 %s t）：%s(%.4f) vs %s(%.4f)，均值差 %.4f，t=%.3f，p=%s，%s；%s=%s（%s效应）" % (
        _fmt_p(lev.get("p_value") if lev else None),
        "方差齐" if equal_var else "方差不齐",
        "Student" if method == "student" else "Welch",
        na_name, m1, nb_name, m2, diff, t, _fmt_p(p),
        "差异显著" if _sig(p) else "差异不显著",
        effect_name, ("%.3f" % d) if d is not None else "NA", _effect_label("d", d))
    if g is not None:
        interp += "；Hedges' g=%.3f" % g
    if gate_dev:
        interp += "；" + gate_dev
    d_ci = _cohen_d_ci(d, n1, n2) if (equal_var and d is not None) else None
    glass_ci = _cohen_d_ci(d, n1, n2) if (not equal_var and d is not None) else None
    php = _posthoc_power_t(d, n1, n2)
    reasons = []
    if min(n1, n2) < 5:
        reasons.append("存在样本量<5的分组（最少 %d 条），统计功效可能不足" % min(n1, n2))
    if php is not None and php < 0.5:
        reasons.append("事后统计功效偏低（=%.2f < 0.5），当前观测效应易被漏检" % php)
    return _ttest_summary(t, df, p, summary_d, ci, {
        "type": "independent", "method": method, "levene": lev,
        "effect_name": effect_name,
        "cohen_d": _r4(d) if equal_var else None,
        "glass_delta": _r4(d) if not equal_var else None,
        "hedges_g": _r4(g),
        "cohen_d_ci": d_ci, "glass_delta_ci": glass_ci,
        "effect_size_label": _effect_label("d", d),
        "posthoc_power": php,
        "bayes_factor_10": bf10,
        "gate_deviation": gate_dev,
        "permutation_p": _r4(perm_p) if perm_p is not None else None,
        "group_a": {"name": na_name, "n": n1, "mean": round(m1, 4), "std": round(math.sqrt(v1), 4)},
        "group_b": {"name": nb_name, "n": n2, "mean": round(m2, 4), "std": round(math.sqrt(v2), 4)},
        "mean_diff": round(diff, 4),
        "low_evidence_advisory": _low_evidence_advisory_text(reasons),
        "interpretation": interp})


def _paired_se(rows, f1, f2):
    """配对样本：收集有效数值对并计算差值序列、均值、标准差、SE、df、n。
    任一前置条件不满足（有效对 <2 或差值标准差为 0）时返回 None。"""
    pairs = [(to_float(r.get(f1)), to_float(r.get(f2))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return None
    diffs = [a - b for a, b in pairs]
    md, sd = mean(diffs), stdev(diffs)
    if not sd or sd != sd:
        return None
    se = sd / math.sqrt(n)
    return diffs, md, sd, se, n - 1, n


def _ttest_paired(rows, spec):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "ttest-paired 需 '字段1,字段2'"}
    f1, f2 = parts
    res = _paired_se(rows, f1, f2)
    if res is None:
        return {"error": "有效配对不足（<2）或差值标准差为 0，无法检验"}
    diffs, md, sd, se, df, n = res
    t = md / se
    p = _t_two_sided_p(abs(t), df)
    sse_alt = sum((x - md) ** 2 for x in diffs)
    sse_null = sum(x * x for x in diffs)
    bf10 = _bf10_bic(n, sse_null, sse_alt, 1, 2)
    d = md / sd
    ci = _diff_ci(md, se, df)
    dz_ci = None
    if d is not None and n > 2:
        se_dz = math.sqrt(1.0 / n + d * d / (2.0 * n))
        zc = _norm_ppf(0.975)
        if zc is not None:
            dz_ci = [round(d - zc * se_dz, 4), round(d + zc * se_dz, 4)]
    interp = "配对样本 t 检验（n=%d 对）：%s−%s 平均差 %.4f，t=%.3f，p=%s，%s（α=%g）；Cohen's dz=%.3f（%s效应）" % (
        n, f1, f2, md, t, _fmt_p(p), "差异显著" if _sig(p) else "差异不显著", _CFG["alpha"],
        d, _effect_label("d", d))
    php = _posthoc_power_t(d, n, n)
    reasons = []
    if n < 5:
        reasons.append("配对样本量偏小（n=%d 对），统计功效可能不足" % n)
    if php is not None and php < 0.5:
        reasons.append("事后统计功效偏低（=%.2f < 0.5），当前观测效应易被漏检" % php)
    return _ttest_summary(t, df, p, d, ci, {
        "type": "paired", "field_a": f1, "field_b": f2, "n_pairs": n,
        "mean_diff": round(md, 4), "std_diff": round(sd, 4),
        "cohen_dz": _r4(d), "cohen_dz_ci": dz_ci,
        "bayes_factor_10": bf10,
        "posthoc_power": php,
        "low_evidence_advisory": _low_evidence_advisory_text(reasons),
        "interpretation": interp})


def _wilcoxon(rows, spec):
    """Wilcoxon 符号秩检验（配对样本非参）：z 近似 + 结校正 + 效应量 r。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "wilcoxon 需 '字段1,字段2'"}
    f1, f2 = parts
    pairs = [(to_float(r.get(f1)), to_float(r.get(f2))) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n_total = len(pairs)
    diffs = [a - b for a, b in pairs]
    n_zero = sum(1 for d in diffs if d == 0)
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n < 1:
        return {"error": "有效非零差值不足（<1），无法检验"}
    abs_diffs = [abs(d) for d in diffs]
    ranks = _ranks(abs_diffs)
    W_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    W_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    ties = Counter(abs_diffs)
    tcorr = sum(t ** 3 - t for t in ties.values() if t > 1)
    mu = n * (n + 1) / 4.0
    sigma_sq = n * (n + 1) * (2 * n + 1) / 24.0 - tcorr / 48.0
    if sigma_sq <= 0:
        return {"error": "方差退化（差值可能全相同），无法计算 z 近似"}
    z = (W_plus - mu) / math.sqrt(sigma_sq)
    p = 2.0 * _norm_sf(abs(z))
    p = min(1.0, p)
    r_effect = abs(z) / math.sqrt(n) if n > 0 else None
    md = sum(diffs) / n
    a = _CFG["alpha"]
    interp = ("Wilcoxon 符号秩检验（配对样本非参）：n=%d 对（排除 %d 个零差值），"
              "W+=%.1f，W-=%.1f，z=%.4f，p=%s，%s（α=%g）；效应量 r=%.3f（%s效应）"
              % (n_total, n_zero, W_plus, W_minus, z, _fmt_p(p),
                 "差异显著" if _sig(p, a) else "差异不显著", a,
                 r_effect if r_effect is not None else float('nan'),
                 _effect_label("d", r_effect) if r_effect is not None else "NA"))
    return {"type": "wilcoxon_signed_rank", "field_a": f1, "field_b": f2,
            "n_pairs": n_total, "n_zeros_excluded": n_zero, "n_effective": n,
            "W_plus": round(W_plus, 1), "W_minus": round(W_minus, 1),
            "z": _r4(z), "p_value": _r4(p),
            "effect_size_r": _r4(r_effect),
            "effect_size_label": _effect_label("d", r_effect) if r_effect is not None else None,
            "significant": bool(_sig(p, a)),
            "interpretation": interp}


def _friedman(rows, spec):
    """Friedman 检验（重复测量非参）：Kendall's W + Dunn 事后比较。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        return {"error": "friedman 需 '受试者,条件,数值'"}
    subj_f, cond_f, val_f = parts
    data = {}
    for r in rows:
        s = str(r.get(subj_f))
        c = str(r.get(cond_f))
        v = to_float(r.get(val_f))
        if v is not None and s and c:
            data.setdefault(s, {})[c] = v
    all_conds = sorted(set(c for s in data for c in data[s]))
    k = len(all_conds)
    subjects = [s for s in data if len(data[s]) == k]
    n = len(subjects)
    if n < 3 or k < 2:
        return {"error": "Friedman 需 ≥3 受试者 × ≥2 条件（完整数据），当前 n=%d, k=%d" % (n, k)}
    rank_sums = [0.0] * k
    rank_sq_sums = [0.0] * k
    for s in subjects:
        vals = [data[s][c] for c in all_conds]
        rks = _ranks(vals)
        for j in range(k):
            rank_sums[j] += rks[j]
            rank_sq_sums[j] += rks[j] ** 2
    chi2 = (12.0 / (n * k * (k + 1))) * sum(r ** 2 for r in rank_sums) - 3 * n * (k + 1)
    df = k - 1
    p = _chi2_sf(chi2, df)
    kendall_w = chi2 / (n * (k - 1)) if n > 0 else None
    a = _CFG["alpha"]
    mean_ranks = [r / n for r in rank_sums]
    interp = ("Friedman 检验（重复测量非参）：n=%d 受试者 × k=%d 条件，"
              "χ²=%.4f（df=%d），p=%s，%s（α=%g）；Kendall's W=%.3f（%s一致性）"
              % (n, k, chi2, df, _fmt_p(p),
                 "差异显著" if _sig(p, a) else "差异不显著", a,
                 kendall_w if kendall_w is not None else float('nan'),
                 _effect_label("eta2", kendall_w) if kendall_w is not None else "NA"))
    result = {"type": "friedman", "subject_field": subj_f, "condition_field": cond_f,
              "value_field": val_f, "n_subjects": n, "k_conditions": k,
              "conditions": all_conds, "mean_ranks": [round(r, 4) for r in mean_ranks],
              "chi2": _r4(chi2), "df": df, "p_value": _r4(p),
              "kendall_w": _r4(kendall_w),
              "significant": bool(_sig(p, a)),
              "interpretation": interp}
    if k >= 2 and p is not None and _sig(p, a):
        se = math.sqrt(k * (k + 1) / (6.0 * n))
        dunn_pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                z_ij = abs(mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0
                p_ij = 2.0 * _norm_sf(z_ij)
                dunn_pairs.append({"pair": "%s vs %s" % (all_conds[i], all_conds[j]),
                                   "z": _r4(z_ij), "p_value": _r4(p_ij),
                                   "significant": _sig(p_ij, a)})
        if len(dunn_pairs) > 1:
            adj = _p_adjust([d["p_value"] or 1.0 for d in dunn_pairs], "bonferroni")
            for d, ap in zip(dunn_pairs, adj):
                d["p_adjusted"] = _r4(ap)
        result["dunn_posthoc"] = dunn_pairs
    return result


def _mcnemar(rows, spec):
    """McNemar 检验（配对二分类）：连续性校正 + 小样本精确二项检验回退。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "mcnemar 需 '字段1,字段2'（二分类 0/1）"}
    f1, f2 = parts
    b = c = 0
    n_both_pos = n_both_neg = 0
    for r in rows:
        v1 = to_float(r.get(f1))
        v2 = to_float(r.get(f2))
        if v1 is None or v2 is None:
            continue
        x1 = 1 if v1 > 0 else 0
        x2 = 1 if v2 > 0 else 0
        if x1 == 1 and x2 == 0:
            b += 1
        elif x1 == 0 and x2 == 1:
            c += 1
        elif x1 == 1 and x2 == 1:
            n_both_pos += 1
        else:
            n_both_neg += 1
    n_disc = b + c
    if n_disc < 1:
        return {"error": "无不一致配对（discordant pairs=0），无法检验"}
    a = _CFG["alpha"]
    if n_disc < 25:
        k = min(b, c)
        p_val = 0.0
        for i in range(k + 1):
            p_val += _binom_pmf(i, n_disc, 0.5)
        p_val = min(1.0, 2.0 * p_val)
        method = "exact_binomial"
        chi2_val = None
        interp = ("McNemar 检验（精确二项检验，n_disc=%d < 25）：b=%d, c=%d，"
                  "p=%s，%s（α=%g）" % (n_disc, b, c, _fmt_p(p_val),
                   "差异显著" if _sig(p_val, a) else "差异不显著", a))
    else:
        chi2_val = (abs(b - c) - 1) ** 2 / n_disc
        p_val = _chi2_sf(chi2_val, 1)
        method = "chi2_continuity"
        interp = ("McNemar 检验（χ² 连续性校正，n_disc=%d ≥ 25）：b=%d, c=%d，"
                  "χ²=%.4f，p=%s，%s（α=%g）" % (n_disc, b, c, chi2_val,
                   _fmt_p(p_val), "差异显著" if _sig(p_val, a) else "差异不显著", a))
    odds_ratio = (b / c) if c > 0 else None
    return {"type": "mcnemar", "field_a": f1, "field_b": f2,
            "b_discordant": b, "c_discordant": c, "n_discordant": n_disc,
            "n_both_positive": n_both_pos, "n_both_negative": n_both_neg,
            "chi2": _r4(chi2_val), "p_value": _r4(p_val),
            "method": method, "odds_ratio": _r4(odds_ratio),
            "significant": bool(_sig(p_val, a)),
            "interpretation": interp}


def _binom_pmf(k, n, p):
    """二项分布 PMF：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)，零依赖。"""
    if k < 0 or k > n:
        return 0.0
    log_coef = 0.0
    for i in range(1, min(k, n - k) + 1):
        log_coef += math.log(n - k + i) - math.log(i)
    log_p = log_coef + k * math.log(p) + (n - k) * math.log(1 - p)
    return math.exp(log_p)


def _chisq(rows, spec):
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) == 2:
        fa, fb = parts
        obs = OrderedDict()
        a_levels, b_levels = [], []
        for r in rows:
            va, vb = str(r.get(fa)), str(r.get(fb))
            if va in ("", "None") or vb in ("", "None"):
                continue
            if va not in a_levels:
                a_levels.append(va)
            if vb not in b_levels:
                b_levels.append(vb)
            obs[(va, vb)] = obs.get((va, vb), 0) + 1
        R, C = len(a_levels), len(b_levels)
        n = sum(obs.values())
        if R < 2 or C < 2 or n == 0:
            return {"error": "列联表至少需 2×2（当前 %d×%d）" % (R, C)}
        row_tot = {a: sum(obs.get((a, b), 0) for b in b_levels) for a in a_levels}
        col_tot = {b: sum(obs.get((a, b), 0) for a in a_levels) for b in b_levels}
        chi2 = 0.0
        low_exp = 0
        yates = False
        obs_matrix = [[obs.get((a, b), 0) for b in b_levels] for a in a_levels]
        table = []
        for a in a_levels:
            line = {"level": a}
            for b in b_levels:
                e = row_tot[a] * col_tot[b] / n
                o = obs.get((a, b), 0)
                if e < 5:
                    low_exp += 1
                if e > 0:
                    chi2 += (o - e) ** 2 / e
                line[b] = {"observed": o, "expected": round(e, 2)}
            table.append(line)
        df = (R - 1) * (C - 1)
        # 2×2 小样本：启用 Yates 连续性校正，降低一类错误膨胀
        if R == 2 and C == 2:
            yates = True
            scipy_ok = False
            if HAS_SCIPY:
                try:
                    cm = _scipy_try(lambda: spstats.chi2_contingency(obs_matrix, correction=True))
                    if cm is not None:
                        chi2, _, df_c, _ = cm
                        df = df_c
                        scipy_ok = True
                except Exception:
                    cm = None
            if not scipy_ok:
                chi2 = 0.0
                for a in a_levels:
                    for b in b_levels:
                        e = row_tot[a] * col_tot[b] / n
                        o = obs.get((a, b), 0)
                        if e > 0:
                            chi2 += (abs(o - e) - 0.5) ** 2 / e
        p = _chi2_sf(chi2, df)
        cv = math.sqrt(chi2 / (n * (min(R, C) - 1))) if n > 0 and min(R, C) > 1 else None
        low_pct = low_exp / (R * C)
        interp = "卡方独立性检验：%s×%s，χ²=%.3f，df=%d，p=%s，两变量%s（α=%g）；Cramér's V=%s（%s关联）" % (
            fa, fb, chi2, df, _fmt_p(p),
            "存在显著关联" if _sig(p) else "未见显著关联", _CFG["alpha"],
            ("%.3f" % cv) if cv is not None else "NA", _effect_label("cramer_v", cv))
        if yates:
            interp += "（2×2 小样本已应用 Yates 连续性校正）"
        if low_pct > 0.2:
            interp += "。注意：%d%% 格子期望频数<5，卡方近似可能不可靠（建议合并类别或用 Fisher 精确检验）" % round(low_pct * 100)
        # 2×2 列联表额外给出比值比(OR)与相对风险(RR)及其 95% CI
        odds_ratio = risk_ratio = fisher_exact = None
        if R == 2 and C == 2:
            or_v, or_ci, or_ok = _odds_ratio_2x2(obs_matrix)
            rr_v, rr_ci, rr_ok = _risk_ratio_2x2(obs_matrix)
            if or_ok:
                odds_ratio = {"value": or_v, "ci95": or_ci}
                interp += ("；OR=%.3f（95%%CI %.3f–%.3f）" % (or_v, or_ci[0], or_ci[1])) if or_ci else ("；OR=%.3f" % or_v)
            if rr_ok:
                risk_ratio = {"value": rr_v, "ci95": rr_ci}
                interp += ("；RR=%.3f（95%%CI %.3f–%.3f）" % (rr_v, rr_ci[0], rr_ci[1])) if rr_ci else ("；RR=%.3f" % rr_v)
            fe_p = _fisher_exact_2x2(obs_matrix)
            if fe_p is not None:
                fisher_exact = {"p_value": _r4(fe_p)}
                interp += "；Fisher 精确检验 p=%s" % _fmt_p(fe_p)
            # 贝叶斯因子 BF10（BIC 近似，df=1）：补充二元 p 值的证据强度解读
            try:
                bf10_2x2 = round(math.exp((chi2 - math.log(n)) / 2.0), 4)
            except Exception:
                bf10_2x2 = None
            if bf10_2x2 is not None:
                interp += "；BF10=%.3f（>1 支持关联，<1 支持独立）" % bf10_2x2
        return {"type": "independence", "field_a": fa, "field_b": fb,
                "chi2": round(chi2, 4), "df": df, "n": n,
                "yates_correction": yates,
                "p_value": _r4(p), "significant": _sig(p),
                "cramer_v": _r4(cv), "effect_size_label": _effect_label("cramer_v", cv),
                "odds_ratio": odds_ratio, "risk_ratio": risk_ratio, "fisher_exact": fisher_exact,
                "bayes_factor_10": bf10_2x2,
                "low_expected_ratio": round(low_pct, 3),
                "contingency": table, "interpretation": interp}
    elif len(parts) == 1:
        fld = parts[0]
        counts = OrderedDict()
        for r in rows:
            v = str(r.get(fld))
            if v in ("", "None"):
                continue
            counts[v] = counts.get(v, 0) + 1
        k = len(counts)
        n = sum(counts.values())
        if k < 2 or n == 0:
            return {"error": "拟合优度检验至少需 2 个类别"}
        e = n / k
        chi2 = sum((o - e) ** 2 / e for o in counts.values())
        df = k - 1
        p = _chi2_sf(chi2, df)
        interp = "卡方拟合优度检验（均匀分布假设）：%s 共 %d 类，χ²=%.3f，df=%d，p=%s，分布%s均匀（α=%g）" % (
            fld, k, chi2, df, _fmt_p(p), "显著偏离" if _sig(p) else "未显著偏离", _CFG["alpha"])
        if e < 5:
            interp += "。注意：期望频数 %.1f<5，结果仅供参考" % e
        return {"type": "goodness_of_fit", "field": fld,
                "chi2": round(chi2, 4), "df": df, "n": n,
                "p_value": _r4(p), "significant": _sig(p),
                "observed": counts, "expected_each": round(e, 2),
                "interpretation": interp}
    return {"error": "chisq 需 '字段A,字段B'（独立性）或 '字段'（拟合优度）"}


def _vif(columns, rows):
    num = _numeric_columns(columns, rows)
    if len(num) < 2:
        return {"error": "VIF 需至少 2 个数值列"}
    series = _numeric_series(rows, num)

    std = {}
    for c in num:
        xs = [x for x in series[c] if x is not None]
        m, s = mean(xs), stdev(xs)
        std[c] = [(x - m) / s if (x is not None and s > 0) else 0.0 for x in series[c]]
    R = {a: {b: pearson(std[a], std[b]) for b in num} for a in num}
    vif = OrderedDict()
    for i, c in enumerate(num):
        others = [b for b in num if b != c]
        if not others:
            vif[c] = 1.0
            continue


        idx = {b: k for k, b in enumerate(others)}
        M = [[R[a_][b_] for b_ in others] for a_ in others]
        rhs = [R[c][b_] for b_ in others]
        beta = gaussian_solve(M, rhs)
        if beta is None:
            vif[c] = float("inf")
            continue
        r2 = sum(beta[k] * R[c][others[k]] for k in range(len(others)))
        if r2 <= 0.0:
            vif[c] = 1.0
        elif r2 >= 1.0 - 1e-12:
            vif[c] = float("inf")
        else:
            vif[c] = round(1.0 / (1.0 - r2), 4)
    high = [c for c, v in vif.items() if v is not None and v == v and v > 5]
    interp = ("存在较强多重共线性的变量：" + "、".join(high) + "（VIF>5），回归建模时建议剔除或合并"
              ) if high else "各变量 VIF 均 ≤5，未见明显多重共线性"
    return {"vif": vif, "note": "VIF>5 表示存在较强多重共线性", "interpretation": interp}


def _rm_anova(rows, spec, alpha=None):
    """单因素重复测量（组内）ANOVA：含 Mauchly 球形检验、Greenhouse-Geisser 校正与配对事后 FDR。
    spec: '受试者,组内因子,数值'（长格式）。零依赖实现。"""
    a = alpha or _CFG["alpha"]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        return {"error": "rm_anova 需 '受试者,组内因子,数值'"}
    sc, wc, vc = parts
    data, levels_order = {}, []
    for r in rows:
        subj = r.get(sc)
        lvl = r.get(wc)
        v = to_float(r.get(vc))
        if subj is None or lvl is None or v is None:
            continue
        lvl = str(lvl)
        data.setdefault(subj, {})[lvl] = v
        if lvl not in levels_order:
            levels_order.append(lvl)
    if len(levels_order) < 2:
        return {"error": "组内因子至少需 2 个水平"}
    subs = [s for s, d in data.items() if all(l in d for l in levels_order)]
    if len(subs) < 3:
        return {"error": "有效配对受试者不足（需 ≥3 且每人覆盖全部水平）"}
    k, n = len(levels_order), len(subs)
    X = [[data[s][lvl] for lvl in levels_order] for s in subs]
    grand = mean([v for row in X for v in row])
    level_means = [mean(X[i][j] for i in range(n)) for j in range(k)]
    subj_means = [mean(X[i]) for i in range(n)]
    ss_tot = sum((X[i][j] - grand) ** 2 for i in range(n) for j in range(k))
    ss_subj = k * sum((m - grand) ** 2 for m in subj_means)
    ss_treat = n * sum((level_means[j] - grand) ** 2 for j in range(k))
    ss_err = ss_tot - ss_subj - ss_treat
    df_treat, df_err = k - 1, (k - 1) * (n - 1)
    ms_treat = ss_treat / df_treat if df_treat else 0.0
    ms_err = ss_err / df_err if df_err else 0.0
    F = ms_treat / ms_err if ms_err > 0 else float("inf")
    p = _f_sf(F, df_treat, df_err) if ms_err > 0 else None
    # 球形度：以正交对照构造 T=XC^T，S=T^T T，GG ε = tr(S)^2/(m·tr(S^2))
    C = _orthonormal_contrast(k)
    m = k - 1
    T = [[sum(X[i][c2] * C[r][c2] for c2 in range(k)) for r in range(m)] for i in range(n)]
    S = [[sum(T[i][r] * T[i][c] for i in range(n)) for c in range(m)] for r in range(m)]
    eig = _eigvalsh_jacobi(S)
    trS = sum(eig)
    trS2 = sum(e * e for e in eig)
    gg_eps = (trS ** 2) / (m * trS2) if trS2 > 0 else 1.0
    gg_eps = min(1.0, max(1.0 / m, gg_eps))
    num_df_c, den_df_c = df_treat * gg_eps, df_err * gg_eps
    p_gg = _f_sf(F, num_df_c, den_df_c) if ms_err > 0 else None
    mauchly, mchi2, mp, spher_ok = None, None, None, None
    if m >= 2:
        detS = _det_sym(S)
        if detS is not None and trS > 0:
            W = detS / ((trS / m) ** m)
            mauchly = W
            df_m = m * (m - 1) // 2
            if W > 0 and df_m > 0:
                fcorr = 1.0 - (2 * m ** 2 + m + 2) / (6.0 * m * n)
                mchi2 = -(n - 1) * math.log(max(W, 1e-300)) * fcorr
                mp = _chi2_sf(mchi2, df_m)
                spher_ok = bool(mp is not None and mp >= a)
    # 配对事后（每对水平配对 t + FDR）
    pairs, raw = [], []
    for i in range(k):
        for j in range(i + 1, k):
            dij = [X[s][i] - X[s][j] for s in range(n)]
            md = mean(dij)
            sd = stdev(dij) if len(dij) > 1 else 0.0
            se = sd / math.sqrt(len(dij)) if len(dij) > 1 else 0.0
            tt = md / se if se > 0 else None
            pij = _t_two_sided_p(abs(tt), len(dij) - 1) if tt is not None else None
            pairs.append((levels_order[i], levels_order[j]))
            raw.append(pij)
    valid = [p for p in raw if p is not None]
    adj = _p_adjust(valid, "fdr")
    posthoc = []
    ai = 0
    for idx, (a1, b1) in enumerate(pairs):
        pij = raw[idx]
        ap = adj[ai] if (pij is not None and ai < len(adj)) else None
        if pij is not None:
            ai += 1
        md = mean([X[s][levels_order.index(a1)] - X[s][levels_order.index(b1)] for s in range(n)])
        posthoc.append({"pair": "%s vs %s" % (a1, b1), "mean_diff": round(md, 4),
                        "p_value": _r4(pij), "adjusted_p_fdr": _r4(ap),
                        "significant": bool(pij is not None and pij < a and (ap is None or ap < a))})
    mtxt = ("球形度假设满足（Mauchly W=%s, p=%s），无需校正" % (_r4(mauchly), _fmt_p(mp))
            if spher_ok else
            "球形度不满足（Mauchly W=%s, p=%s），已按 Greenhouse-Geisser ε=%.3f 校正自由度" % (_r4(mauchly), _fmt_p(mp), gg_eps))
    return {"type": "rm_anova", "n_subjects": n, "n_levels": k, "levels": levels_order,
            "sphericity": {"mauchly_W": _r4(mauchly), "chi2": _r4(mchi2),
                           "df": (m * (m - 1) // 2) if m >= 2 else None,
                           "p_value": _r4(mp), "assumed": spher_ok},
            "greenhouse_geisser_epsilon": _r4(gg_eps),
            "anova": {"ss_treatment": round(ss_treat, 4), "ss_subjects": round(ss_subj, 4),
                      "ss_error": round(ss_err, 4), "ss_total": round(ss_tot, 4),
                      "df_treatment": df_treat, "df_error": df_err,
                      "F": round(F, 4), "p_value": _r4(p), "significant": _sig(p),
                      "F_gg_corrected": round(F, 4), "df1_gg": round(num_df_c, 3),
                      "df2_gg": round(den_df_c, 3), "p_gg": _r4(p_gg),
                      "significant_gg": _sig(p_gg)},
            "posthoc_paired_fdr": posthoc,
            "interpretation": "重复测量（组内）ANOVA：F(%.3f,%.3f)=%.4f，p_GG=%s；%s" % (
                num_df_c, den_df_c, F, _fmt_p(p_gg), mtxt)}


def cmd_stats(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    if getattr(args, "check_preregister", None):
        _check_preregister(args.check_preregister, {
            "input": args.input,
            "alpha": _CFG["alpha"],
            "ss_type": getattr(args, "ss_type", "III"),
            "adjust": getattr(args, "adjust", None),
        })
    result = {"columns": columns,
              "config": {
                  "input": args.input,
                  "alpha": _CFG["alpha"],
                  "shapiro_max_n": _CFG["shapiro_max_n"],
                  "ss_type": getattr(args, "ss_type", None),
                  "adjust": (getattr(args, "adjust", None) or "none"),
                  "discipline": getattr(args, "discipline", None),
                  "tests": {k: getattr(args, k) for k in
                            ("describe", "corr", "normality", "anova", "ttest_one",
                             "ttest", "ttest_paired", "mannwhitney",
                             "wilcoxon", "friedman", "mcnemar",
                             "chisq", "tukey", "vif")
                            if getattr(args, k, None)},
              }}
    if args.describe:
        result["describe"] = _describe(columns, rows)
    if args.corr:
        result["correlation"] = _corr_matrix(columns, rows,
                                             permutation=bool(args.permutation),
                                             n_perm=args.n_perm, seed=args.seed)
    if args.normality:
        result["normality"] = _normality(columns, rows)
    if args.anova:
        result["anova"] = _anova(columns, rows, args.anova, args.ss_type)
    if args.ttest_one:
        result["ttest_one_sample"] = _ttest_one_sample(rows, args.ttest_one)
    if args.ttest:
        result["ttest_independent"] = _ttest_independent(rows, args.ttest,
                                                         permutation=bool(args.permutation),
                                                         n_perm=args.n_perm, seed=args.seed)
    if args.ttest_paired:
        result["ttest_paired"] = _ttest_paired(rows, args.ttest_paired)
    if args.mannwhitney:
        result["mann_whitney"] = _mann_whitney(rows, args.mannwhitney, args.alpha)
    if args.wilcoxon:
        result["wilcoxon"] = _wilcoxon(rows, args.wilcoxon)
    if args.friedman:
        result["friedman"] = _friedman(rows, args.friedman)
    if args.mcnemar:
        result["mcnemar"] = _mcnemar(rows, args.mcnemar)
    if args.chisq:
        result["chisq"] = _chisq(rows, args.chisq)
    if args.tukey:
        result["tukey"] = _tukey(columns, rows, args.tukey)
    if args.vif:
        result["vif"] = _vif(columns, rows)
    if args.rm_anova:
        result["rm_anova"] = _rm_anova(rows, args.rm_anova, args.alpha)
    # ---- T2⑤ / T3⑤ 扩展：事后检验、信度、协变量、混合、中介 ----
    if args.dunnett:
        parts = [p.strip() for p in args.dunnett.split(",")]
        if len(parts) != 2 or not args.control:
            die("Dunnett 需 'factor,value' 且指定 --control 对照水平")
        grp = _group_values(rows, parts[0], parts[1])
        result["dunnett"] = _dunnett(grp, args.control, _CFG["alpha"])
    if args.nemenyi:
        parts = [p.strip() for p in args.nemenyi.split(",")]
        grp = _group_values(rows, parts[0], parts[1])
        result["nemenyi"] = _nemenyi(grp, _CFG["alpha"])
    if args.scheffe:
        parts = [p.strip() for p in args.scheffe.split(",")]
        grp = _group_values(rows, parts[0], parts[1])
        pairs = [(k, v) for k, vs in grp.items() for v in vs]
        owa = _one_way_anova(pairs)
        mse = (owa.get("SSE") or 0.0)
        dfe = (owa.get("n") or 0) - (owa.get("factor_levels") or 0)
        mse = mse / dfe if dfe > 0 and mse else None
        result["scheffe"] = _scheffe(grp, mse, _CFG["alpha"])
    if args.compact_letters:
        parts = [p.strip() for p in args.compact_letters.split(",")]
        grp = _group_values(rows, parts[0], parts[1])
        labels = list(grp.keys())
        sig = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                _, p = _t_test_two(grp[labels[i]], grp[labels[j]])
                sig[(labels[i], labels[j])] = bool(p is not None and p < _CFG["alpha"])
        means = {l: mean(grp[l]) for l in labels}
        result["compact_letters"] = _compact_letters(labels, [means[l] for l in labels], sig)
    if args.icc:
        parts = [p.strip() for p in args.icc.split(",")]
        if len(parts) != 3:
            die("icc 需 'subject,rater,value'")
        subj, rater, val = parts
        d = OrderedDict()
        for r in rows:
            s, rt, v = r.get(subj), r.get(rater), to_float(r.get(val))
            if s is not None and rt is not None and v is not None:
                d.setdefault(str(s), {})[str(rt)] = v
        raters = sorted({rt for m in d.values() for rt in m})
        matrix = [[m.get(rt, 0.0) for rt in raters] for m in d.values()]
        result["icc"] = _icc(matrix, args.icc_model)
    if args.cronbach:
        items = [c.strip() for c in args.cronbach.split(",")]
        for c in items:
            if c not in columns:
                die("Cronbach 题项不存在: %s" % c)
        matrix = [[to_float(r.get(c)) for c in items] for r in rows
                  if all(to_float(r.get(c)) is not None for c in items)]
        result["cronbach"] = _cronbach_alpha(matrix)
    if args.ancova:
        parts = [p.strip() for p in args.ancova.split(",")]
        if len(parts) != 3:
            die("ancova 需 'group,covariate,value'")
        result["ancova"] = _ancova(parts[0], parts[1], parts[2], rows)
    if args.mixed:
        parts = [p.strip() for p in args.mixed.split(",")]
        if len(parts) != 4:
            die("mixed 需 'within,between,subject,value'")
        result["mixed_anova"] = _mixed_anova(parts[0], parts[1], parts[2], parts[3], rows)
    if args.mediation:
        parts = [p.strip() for p in args.mediation.split(",")]
        if len(parts) != 3:
            die("mediation 需 'x,m,y'")
        result["mediation"] = _mediation(parts[0], parts[1], parts[2], rows)
    if args.moderation:
        result["moderation"] = _moderation(rows, args.moderation)
    if args.bootstrap_mediation:
        result["bootstrap_mediation"] = _bootstrap_mediation(
            rows, args.bootstrap_mediation, args.n_boot, args.seed)
    if args.stepwise:
        result["stepwise"] = _stepwise(rows, args.stepwise, args.direction, args.entry_p)
    if len(result) == 1:
        result["describe"] = _describe(columns, rows)
    # 多主检验家族wise 控制：对同时执行的若干检验的 p 值做多重比较校正
    if len(result) > 1:
        fam = []
        mapping = []
        for key in ("anova", "ttest_independent", "ttest_paired", "ttest_one_sample",
                     "chisq", "wilcoxon", "friedman", "mcnemar"):
            if key in result:
                p = result[key].get("p_value")
                if p is not None and p == p:
                    fam.append(p); mapping.append(key)
        if mapping:
            method = args.adjust or "none"
            if method in ("holm", "bonferroni", "fdr"):
                adj = _p_adjust(fam, method)
                for key, ap in zip(mapping, adj):
                    result[key]["adjusted_p"] = _r4(ap)
                    result[key]["adjust_method"] = method
                result["multiple_comparison"] = {
                    "tests": mapping, "method": method,
                    "note": "已对同时执行的 %d 个主检验的 p 值做 %s 多重比较校正，adjusted_p 为校正后值" % (len(mapping), method)}
            else:
                result["multiple_comparison"] = {
                    "tests": mapping, "method": "none",
                    "note": "本次同时执行 %d 个主检验；p 值未做家族wise 校正，多次检验下假阳性率上升，建议加 --adjust holm/bonferroni/fdr" % len(mapping)}
    emit({"status": "ok", "task": "stats", "result": result})

def cmd_report(args):
    columns, rows = load_rows(args.input)
    numeric_cols = _numeric_columns(columns, rows)
    clean_r = _describe(columns, rows)
    disc = None
    if getattr(args, "discipline", None):
        disc = _discipline_index(args.discipline)
        if disc == "auto":
            disc = _discipline_index(infer_discipline(columns)["key"])
        elif disc is None:
            die("未知学科: %s（可选: %s / auto）" % (
                args.discipline, ", ".join(d["key"] for d in DISCIPLINE_REGISTRY)))
    anom = OrderedDict()
    rules_used = set()
    for c in numeric_cols:
        series = _col_floats(rows, c)
        if disc:
            r = _discipline_anomaly(series, disc, c, rules_used)
        else:
            r = _anomaly_column(series, "iqr", "L2")
        anom[c] = r
    anom_out = {}
    for c, v in anom.items():
        e = {"count": v["count"], "used": v.get("used")}
        if disc:
            e["discipline_rule"] = v.get("discipline_rule")
            e["advice"] = v.get("advice")
        anom_out[c] = e
    rep = {
        "task": "report",
        "run_id": "run-%s-%s" % (int(__import__("time").time()), abs(hash(args.input)) % 100000),
        "environment": {
            "python_version": __import__("platform").python_version(),
            "numpy": ("%s (加速)" % np.__version__) if HAS_NUMPY else "未安装（零依赖回退）",
            "scipy": "已安装" if HAS_SCIPY else "未安装（零依赖回退）",
            "sklearn": "已安装" if HAS_SKLEARN else "未安装（零依赖回退）",
            "zero_dependency_core": (not HAS_NUMPY and not HAS_SCIPY and not HAS_SKLEARN),
        },
        "dataset": {
            "rows": len(rows), "columns": columns, "numeric_columns": numeric_cols,
        },
        "summary": clean_r,
        "anomaly_iqr_L2": anom_out,
        "provenance": "由 BaiChuanShuHui Skill (skill_runtime.py) 自动生成",
    }
    if disc:
        rep["discipline"] = {"key": disc["key"], "name": disc["name"],
                             "rules_applied": sorted(rules_used)}
    if args.output:
        _atomic_write(args.output, json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        rep["output"] = args.output
    emit({"status": "ok", "result": rep})

def _mice_impute(rows, columns, numeric_cols, sentinels, iters=5, seed=0):
    """零依赖链式方程多重插补（MICE）：对每个含缺失的数值列，以其余数值列为预测变量做 OLS 迭代插补。"""
    sentinels = set(sentinels)
    data = {c: [] for c in numeric_cols}
    missing = {c: set() for c in numeric_cols}
    for i, r in enumerate(rows):
        for c in numeric_cols:
            v = to_float(r.get(c))
            if v is None or v in sentinels:
                data[c].append(None); missing[c].add(i)
            else:
                data[c].append(v)
    total = sum(len(s) for s in missing.values())
    if total == 0:
        return 0
    rng = random.Random(seed)
    for c in numeric_cols:
        vals = [v for v in data[c] if v is not None]
        fill = mean(vals) if vals else 0.0
        for i in range(len(data[c])):
            if data[c][i] is None:
                data[c][i] = fill
    others = {c: [x for x in numeric_cols if x != c] for c in numeric_cols}
    for _ in range(iters):
        for c in numeric_cols:
            if not missing[c]:
                continue
            oc = others[c]
            p = len(oc) + 1
            X, y = [], []
            for i in range(len(data[c])):
                if i in missing[c]:
                    continue
                xi = [1.0] + [data[o][i] for o in oc]
                if all(v is not None for v in xi):
                    X.append(xi); y.append(data[c][i])
            if len(X) < p + 1:
                continue
            beta = None
            if HAS_NUMPY:
                try:
                    Xm = np.asarray(X, dtype=float)
                    yv = np.asarray(y, dtype=float)
                    beta_arr = np.linalg.lstsq(Xm, yv, rcond=None)[0]
                    beta = [float(b) for b in beta_arr]
                except Exception:
                    beta = None
            if beta is None:
                XtX = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]
                Xty = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
                beta = gaussian_solve(XtX, Xty)
            if beta is None:
                continue
            for i in missing[c]:
                xi = [1.0] + [data[o][i] for o in oc]
                pred = sum(beta[j] * xi[j] for j in range(p))
                data[c][i] = pred + (rng.random() - 0.5) * 0.01 * abs(pred if pred else 1.0)
    fixed = 0
    for i, r in enumerate(rows):
        for c in numeric_cols:
            if i in missing[c]:
                r[c] = round(data[c][i], 6)
                fixed += 1
    return fixed


def cmd_missing(args):
    _apply_cfg(args)
    sentinels = _CFG["sentinels"]
    columns, rows = load_rows(args.input)
    n = len(rows)
    numeric_cols = _numeric_columns(columns, rows)
    col_report = []
    sentinel_findings = []
    for c in columns:
        miss = sum(1 for r in rows if _is_missing(r.get(c)))
        sentinel = OrderedDict()
        if c in numeric_cols:
            vals = _col_floats(rows, c)
            for sv in sentinels:
                cnt = sum(1 for v in vals if v is not None and v == sv)
                if cnt > 0:
                    key = str(int(sv)) if float(sv).is_integer() else str(sv)
                    sentinel[key] = cnt
                    sentinel_findings.append({"field": c, "sentinel": key, "count": cnt})
        col_report.append({
            "name": c,
            "type": "numeric" if c in numeric_cols else "text",
            "missing": miss,
            "missing_pct": round(miss / n * 100, 1) if n else 0.0,
            "sentinel": sentinel,
        })

    m_rows = min(args.sample_rows, n)
    matrix = {c: [0 if _is_missing(rows[i].get(c)) else 1
                  for i in range(m_rows)] for c in columns}

    output = None
    fixed = 0
    if args.fix_sentinel:
        if args.strategy == "mice":
            fixed = _mice_impute(rows, columns, numeric_cols, sentinels, iters=5, seed=args.seed or 0)
        else:
            for c in numeric_cols:
                iv = [(i, to_float(r.get(c))) for i, r in enumerate(rows)]
                hit = [i for i, v in iv if v is not None and v in sentinels]
                if not hit:
                    continue
                if args.strategy in ("mean", "median"):
                    good = [v for i, v in iv if v is not None and v not in sentinels]
                    fill = (mean(good) if args.strategy == "mean" else median(good)) if good else None
                else:
                    fill = None
                for i in hit:
                    rows[i][c] = fill
                    fixed += 1
        if args.output:
            _write_rows_csv(args.output, rows, columns)
            output = args.output
    emit({
        "status": "ok", "task": "missing",
        "result": {
            "total_rows": n, "total_cols": len(columns),
            "columns": col_report,
            "sentinel_findings": sentinel_findings,
            "sentinels_checked": [str(int(s)) if float(s).is_integer() else str(s)
                                  for s in sentinels],
            "missing_rule": "None/空白串/NA词表(%s)" % ",".join(sorted(_MISSING_TOKENS)),
            "matrix": matrix, "matrix_rows": m_rows,
            "fixed_sentinel_cells": fixed if args.fix_sentinel else None,
            "strategy": args.strategy if args.fix_sentinel else None,
            "output": output,
        },
    })

def cmd_quality(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    n = len(rows)
    numeric_cols = _numeric_columns(columns, rows)
    total_cells = n * len(columns)
    missing = sum(1 for r in rows for c in columns if _is_missing(r.get(c)))
    completeness = 1.0 - (missing / total_cells if total_cells else 0.0)

    anom_ratios = []
    non_finite = 0
    for c in numeric_cols:
        xs = _col_floats(rows, c, dropna=True)
        non_finite += sum(1 for x in xs if math.isinf(x))
        if len(xs) >= 4:
            q1, q3, lo, hi = _iqr_bounds(xs, 1.5)
            if q3 - q1 > 0:
                anom_ratios.append(sum(1 for x in xs if x < lo or x > hi) / len(xs))
    anomaly_rate = mean(anom_ratios) if anom_ratios else 0.0

    norm_pass, norm_detail = 0, OrderedDict()
    for c in numeric_cols:
        xs = _col_floats(rows, c, dropna=True)
        nr = _normality_one(xs)
        if nr.get("normal") is None:
            continue
        norm_detail[c] = nr
        if nr["normal"]:
            norm_pass += 1
    normality_rate = (norm_pass / len(norm_detail)) if norm_detail else 1.0

    max_vif = 0.0
    if len(numeric_cols) >= 2:
        vd = _vif(columns, rows)
        vals = [v for v in (vd.get("vif") or {}).values()
                if isinstance(v, (int, float)) and v == v and v != float("inf")]
        max_vif = max(vals) if vals else 0.0
    collinearity_penalty = min(1.0, max(0.0, (max_vif - 5) / 10)) if max_vif > 5 else 0.0

    s_comp = completeness * 40
    s_anom = (1 - min(anomaly_rate / 0.1, 1.0)) * 30
    s_norm = normality_rate * 20
    s_coll = (1 - collinearity_penalty) * 10
    total = round(s_comp + s_anom + s_norm + s_coll, 1)
    grade = "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D"
    suggestions = []
    if completeness < 0.9:
        suggestions.append("完整度仅 %.1f%%，建议检查缺失机制并选择合适的填充或剔除策略。" % (completeness * 100))
    if anomaly_rate > 0.05:
        suggestions.append("异常占比 %.1f%% 偏高，建议复核是否为仪器误差或真实极端样本。" % (anomaly_rate * 100))
    if normality_rate < 0.6:
        suggestions.append("仅 %.1f%% 数值字段近似正态，非参数检验或正态化变换（skew 子命令）可能更合适。" % (normality_rate * 100))
    if max_vif > 5:
        suggestions.append("存在 VIF=%.1f 的高度共线性变量，建议剔除冗余特征或做降维。" % max_vif)
    if non_finite:
        suggestions.append("检测到 %d 个非有限数值(inf)，会污染均值/方差/相关等统计量，建议核查数据来源或以 --reject-inf 按缺失处理。" % non_finite)
    if not suggestions:
        suggestions.append("数据质量良好，可直接进入统计分析与可视化。")
    emit({
        "status": "ok", "task": "quality",
        "result": {
            "score": total, "grade": grade,
            "dimensions": {
                "completeness": round(completeness, 4),
                "anomaly_rate": round(anomaly_rate, 4),
                "normality_rate": round(normality_rate, 4),
                "max_vif": round(max_vif, 3),
                "non_finite": non_finite,
            },
            "sub_scores": {
                "completeness": round(s_comp, 1), "anomaly": round(s_anom, 1),
                "distribution": round(s_norm, 1), "collinearity": round(s_coll, 1),
            },
            "normality_detail": norm_detail,
            "suggestions": suggestions,
        },
    })

def cmd_skew(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    if args.field not in columns:
        die("字段不存在: %s" % args.field)
    raw = [to_float(r.get(args.field)) for r in rows]
    valid = [x for x in raw if x is not None]
    if len(valid) < 3:
        die("字段 %s 有效数值不足 3 个" % args.field)
    before = _normality_one(valid, with_kurtosis=True)
    method = args.method
    new_field = "%s__%s" % (args.field, method)
    info = {"method": method, "field": args.field, "new_field": new_field}
    if method == "log":
        shift = (-min(valid) + 1.0) if min(valid) <= 0 else 0.0
        if shift:
            info["shift"] = round(shift, 4)
        transformed = [None if x is None else math.log(x + shift) for x in raw]
        info["note"] = "要求数据>0，已自动平移" if shift else "对数变换"
    elif method == "sqrt":
        shift = -min(valid) if min(valid) < 0 else 0.0
        if shift:
            info["shift"] = round(shift, 4)
        transformed = [None if x is None else math.sqrt(x + shift) for x in raw]
        info["note"] = "要求数据≥0，已自动平移" if shift else "平方根变换"
    elif method == "boxcox":
        if not HAS_SCIPY:
            die("Box-Cox 变换需要 scipy（pip install scipy），或改用零依赖的 log/sqrt")
        if min(valid) <= 0:
            die("Box-Cox 要求数据严格为正，请先处理缺失/哨兵值或改用 log/johnson")
        lam = args.lmbda
        if lam is None:
            tvals, lam = spstats.boxcox(valid)
            info["lambda"] = round(float(lam), 4)
        else:
            tvals = spstats.boxcox(valid, lmbda=lam)
            info["lambda"] = lam
        it = iter([float(v) for v in tvals])
        transformed = [None if x is None else next(it) for x in raw]
        info["note"] = "Box-Cox 变换 (λ=%s)" % info["lambda"]
    elif method == "johnson":
        if not HAS_SCIPY:
            die("Johnson 变换需要 scipy（pip install scipy），或改用零依赖的 log/sqrt")
        ja, jb, jloc, jscale = spstats.johnsonsu.fit(valid)
        tvals = spstats.norm.ppf(
            [min(max(spstats.johnsonsu.cdf(v, ja, jb, jloc, jscale), 1e-10), 1 - 1e-10)
             for v in valid])
        it = iter([float(v) for v in tvals])
        transformed = [None if x is None else next(it) for x in raw]
        info["note"] = "Johnson Su 变换 (γ=%s)" % round(float(ja), 3)
    else:
        die("未知变换方法: %s" % method)
    after = _normality_one([v for v in transformed if v is not None], with_kurtosis=True)
    if before.get("shapiro_p") is not None and after.get("shapiro_p") is not None:
        improved = after["shapiro_p"] > before["shapiro_p"]
    else:
        improved = abs(after.get("skewness", 0) or 0) < abs(before.get("skewness", 0) or 0)
    output = None
    if args.output:
        out_cols = columns + [new_field]
        out_rows = []
        for i, r in enumerate(rows):
            row = {c: ("" if r.get(c) is None else r.get(c)) for c in columns}
            row[new_field] = "" if transformed[i] is None else round(transformed[i], 6)
            out_rows.append(row)
        _write_rows_csv(args.output, out_rows, out_cols)
        output = args.output
    preview = [{args.field: (None if raw[i] is None else round(raw[i], 4)),
                new_field: (None if transformed[i] is None else round(transformed[i], 4))}
               for i in range(min(10, len(raw)))]
    emit({
        "status": "ok", "task": "skew",
        "result": {
            "info": info,
            "before": before, "after": after, "improved": bool(improved),
            "preview": preview,
            "output": output,
            "methods_available": ["log", "sqrt"] + (["boxcox", "johnson"] if HAS_SCIPY else []),
        },
    })

VIZ_PALETTE = ["#5b8def", "#3ac5a0", "#e07a5f", "#a06bd6", "#46b1c9", "#d6a23a"]
VIZ_KINDS = ["bar", "hist", "box", "scatter", "line", "grouped", "stacked",
             "pie", "heatmap", "bubble", "sankey", "network", "volcano", "density",
             "errorbar", "mosaic", "km", "forest", "roc", "ridge", "kriging",
             "contour", "flow", "umap", "ma", "multiline"]

VIZ_ROLES = {
    "hist": ["x"], "box": ["x"], "bar": ["x"], "density": ["x"], "ridge": ["x"],
    "scatter": ["x", "y"], "umap": ["x", "y"], "ma": ["x", "y"], "bubble": ["x", "y"],
    "volcano": ["x", "y"], "forest": ["x", "y"], "roc": ["x", "y"], "flow": ["x", "y"],
    "km": ["x", "y"], "errorbar": ["x", "y"], "line": ["x", "y"], "multiline": ["x", "y"],
    "pie": ["x", "y", "z"], "mosaic": ["x", "y", "z"],
    "kriging": ["x", "y", "z"], "contour": ["x", "y", "z"],
    "grouped": ["x", "y", "z"], "stacked": ["x", "y", "z"], "sankey": ["x", "y", "z"],
    "network": ["x"],
    "heatmap": [],
}
VIZ_HTML = """<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>__TITLE_HTML__</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>html,body{margin:0;height:100%;font-family:-apple-system,"Microsoft YaHei",sans-serif;background:__BG__;color:__FG__}#c{width:100vw;height:100vh}</style>
</head>
<body><div id="c"></div>
<script>
var opt=__OPTION__;
var TITLE=__TITLE_JS__;
var chart=echarts.init(document.getElementById('c'),__THEME__);
chart.setOption(opt);
window.addEventListener('resize',function(){chart.resize();});
// 浏览器原生导出（SVG/PNG）
function exportImg(t){var d=chart.getDataURL({type:t,backgroundColor:'__BG__',pixelRatio:2});var a=document.createElement('a');a.href=d;a.download=TITLE+'.'+t;a.click();}
document.addEventListener('keydown',function(e){if(e.key==='s'&&e.ctrlKey){e.preventDefault();exportImg('png');}if(e.key==='e'&&e.ctrlKey){e.preventDefault();exportImg('svg');}});
</script></body></html>"""


def _viz_bar(columns, rows, a):
    xs = [str(r.get(a.x)) for r in rows if r.get(a.x) not in (None, "")]
    cats = sorted(set(xs))
    cnt = [xs.count(c) for c in cats]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": cats}, "yAxis": {"type": "value"},
            "series": [{"type": "bar", "data": cnt, "itemStyle": {"borderRadius": [4, 4, 0, 0]}}]}


def _viz_hist(columns, rows, a):
    xs = [to_float(r.get(a.x)) for r in rows]
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"title": {"text": "无数值数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    lo, hi = min(xs), max(xs)
    if hi == lo:
        hi += 1
    n = max(2, a.bins)
    w = (hi - lo) / n
    bins = [0] * n
    for v in xs:
        i = min(n - 1, int((v - lo) / w))
        bins[i] += 1
    centers = [round(lo + (i + 0.5) * w, 4) for i in range(n)]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": centers, "name": a.x},
            "yAxis": {"type": "value", "name": "频数"},
            "series": [{"type": "bar", "data": bins, "barCategoryGap": "2%"}]}


def _viz_box(columns, rows, a):
    xs = sorted(x for x in (to_float(r.get(a.x)) for r in rows) if x is not None)
    if len(xs) < 4:
        return {"title": {"text": "样本不足"}, "xAxis": {}, "yAxis": {}, "series": []}
    q1, q3, lo, hi = _iqr_bounds(xs, 1.5)
    inl = [v for v in xs if lo <= v <= hi]
    outl = [v for v in xs if v < lo or v > hi]
    box = [inl[0], q1, median(xs), q3, inl[-1]]
    return {"color": VIZ_PALETTE,
            "xAxis": {"type": "category", "data": [a.x]},
            "yAxis": {"type": "value"},
            "series": [
                {"type": "boxplot", "data": [box]},
                {"type": "scatter", "data": [[0, v] for v in outl], "symbolSize": 6}
            ]}


def _viz_scatter(columns, rows, a):
    data = [[to_float(r.get(a.x)), to_float(r.get(a.y))] for r in rows]
    data = [d for d in data if d[0] is not None and d[1] is not None]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.y, "scale": True},
            "series": [{"type": "scatter", "data": data, "symbolSize": 8}]}


def _viz_line(columns, rows, a):
    pts = [(to_float(r.get(a.x)), to_float(r.get(a.y)), r.get(a.z)) for r in rows]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if a.z:
        groups = {}
        for x, y, z in pts:
            groups.setdefault(str(z), []).append((x, y))
        series = [{"type": "line", "name": g,
                   "data": sorted(v, key=lambda t: t[0])} for g, v in sorted(groups.items())]
    else:
        series = [{"type": "line", "data": sorted(pts, key=lambda t: t[0])}]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "legend": {"show": bool(a.z)},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.y}, "series": series}


def _agg_group(rows, a):
    d = {}
    for r in rows:
        xv, yv, zv = r.get(a.x), to_float(r.get(a.y)), r.get(a.z)
        if yv is None or xv in (None, "") or zv in (None, ""):
            continue
        d.setdefault((str(xv), str(zv)), []).append(yv)
    xs = sorted({k[0] for k in d})
    zs = sorted({k[1] for k in d})
    series = []
    for z in zs:
        series.append({"name": z, "type": "bar",
                       "data": [round(mean(d.get((x, z), [])), 3) if d.get((x, z)) else None for x in xs]})
    return xs, series


def _bar_option(xs, series, a):
    """分组/堆叠条形图共用 ECharts option。"""
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "legend": {"show": True}, "xAxis": {"type": "category", "data": xs},
            "yAxis": {"type": "value", "name": a.y}, "series": series}


def _viz_grouped(columns, rows, a):
    xs, series = _agg_group(rows, a)
    return _bar_option(xs, series, a)


def _viz_stacked(columns, rows, a):
    xs, series = _agg_group(rows, a)
    for s in series:
        s["stack"] = "total"
    return _bar_option(xs, series, a)


def _viz_pie(columns, rows, a):
    agg = {}
    for r in rows:
        xv, yv = r.get(a.x), to_float(r.get(a.y))
        if xv in (None, "") or yv is None:
            continue
        agg[str(xv)] = agg.get(str(xv), 0) + yv
    data = [{"name": k, "value": round(v, 4)} for k, v in sorted(agg.items())]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"show": True},
            "series": [{"type": "pie", "radius": "55%" if not a.ring else ["40%", "70%"], "data": data}]}


def _viz_heatmap(columns, rows, a):
    if a.z:
        d = {}
        for r in rows:
            xv, yv, zv = r.get(a.x), to_float(r.get(a.y)), r.get(a.z)
            if yv is None or xv in (None, "") or zv in (None, ""):
                continue
            d.setdefault((str(xv), str(zv)), []).append(yv)
        xs = sorted({k[0] for k in d})
        zs = sorted({k[1] for k in d})
        mat = [[round(mean(d.get((x, z), [])), 4) if d.get((x, z)) else None for z in zs] for x in xs]
        data = [[i, j, mat[i][j]] for i in range(len(xs)) for j in range(len(zs)) if mat[i][j] is not None]
        vals = [v for _, _, v in data]
    else:
        numc = _numeric_columns(columns, rows)
        xs = numc
        mat = [[round(pearson([r.get(ci) for r in rows], [r.get(cj) for r in rows]), 4) for cj in numc] for ci in numc]
        data = [[i, j, mat[i][j]] for i in range(len(numc)) for j in range(len(numc))]
        zs = numc
        vals = [v for _, _, v in data if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    return {"color": VIZ_PALETTE,
            "tooltip": {"position": "top"},
            "grid": {"height": "70%", "top": "10%"},
            "xAxis": {"type": "category", "data": zs, "splitArea": {"show": True}},
            "yAxis": {"type": "category", "data": xs, "splitArea": {"show": True}},
            "visualMap": {"min": vmin, "max": vmax, "calculable": True, "orient": "horizontal", "left": "center", "bottom": "0%"},
            "series": [{"type": "heatmap", "data": data, "label": {"show": False}}]}


def _viz_bubble(columns, rows, a):
    data = [[to_float(r.get(a.x)), to_float(r.get(a.y)), to_float(r.get(a.z))] for r in rows]
    data = [d for d in data if all(v is not None for v in d)]
    zs = [d[2] for d in data] or [1]
    zmin, zmax = min(zs), max(zs)
    rng = (zmax - zmin) or 1
    sz = [8 + 22 * (d[2] - zmin) / rng for d in data]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.y, "scale": True},
            "series": [{"type": "scatter", "data": [list(d) for d in data],
                        "symbolSize": [sz[i] for i in range(len(sz))], "label": {"show": False}}]}


def _viz_sankey(columns, rows, a):
    agg = {}
    for r in rows:
        s, v, t = r.get(a.x), to_float(r.get(a.y)), r.get(a.z)
        if s in (None, "") or t in (None, "") or v is None:
            continue
        agg[(str(s), str(t))] = agg.get((str(s), str(t)), 0) + v
    nodes = []
    seen = set()
    for (s, t) in agg:
        for n in (s, t):
            if n not in seen:
                seen.add(n)
                nodes.append({"name": n})
    links = [{"source": s, "target": t, "value": round(v, 4)} for (s, t), v in agg.items()]
    return {"color": VIZ_PALETTE,
            "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
            "series": [{"type": "sankey", "data": nodes, "links": links,
                        "emphasis": {"focus": "adjacency"}, "lineStyle": {"color": "gradient", "opacity": .55}}]}


def _viz_network(columns, rows, a):
    key = a.z or a.x
    vals = [str(r.get(key)) for r in rows if r.get(key) not in (None, "")]
    seen = set()
    nodes = []
    for i, v in enumerate(vals):
        if v not in seen:
            seen.add(v)
            nodes.append({"name": v, "category": 0,
                         "itemStyle": {"color": VIZ_PALETTE[i % len(VIZ_PALETTE)]}})
    return {"color": VIZ_PALETTE,
            "tooltip": {"trigger": "item"},
            "series": [{"type": "graph", "layout": "force", "roam": True, "draggable": True,
                        "data": nodes, "links": [], "categories": [{"name": key}],
                        "label": {"show": True, "position": "right"},
                        "force": {"repulsion": 160, "edgeLength": [40, 120]}}]}


def _viz_volcano(columns, rows, a):
    sig, nsig = [], []
    for r in rows:
        x, y = to_float(r.get(a.x)), to_float(r.get(a.y))
        if x is None or y is None:
            continue
        (sig if (abs(x) > 1 and y > 1.301) else nsig).append([x, y])
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.y, "scale": True},
            "series": [
                {"type": "scatter", "name": "不显著", "data": nsig, "symbolSize": 7, "itemStyle": {"color": "#999"}},
                {"type": "scatter", "name": "显著(p<0.05)", "data": sig, "symbolSize": 9, "itemStyle": {"color": "#c0392b"}}
            ]}


def _viz_density(columns, rows, a):
    xs = [to_float(r.get(a.x)) for r in rows]
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"title": {"text": "无数值数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    if HAS_SCIPY and HAS_NUMPY:
        from scipy.stats import gaussian_kde
        k = gaussian_kde(np.array(xs))
        lo, hi = min(xs), max(xs)
        grid = [lo + (hi - lo) * i / 199 for i in range(200)]
        ys = [float(np.asarray(k(g)).item()) for g in grid]
    else:
        n = 30
        lo, hi = min(xs), max(xs)
        w = (hi - lo) / n or 1
        cnt = [0] * n
        for v in xs:
            cnt[min(n - 1, int((v - lo) / w))] += 1
        tot = sum(cnt) * w or 1
        grid = [lo + (i + .5) * w for i in range(n)]
        ys = [c / tot for c in cnt]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "value", "name": a.x}, "yAxis": {"type": "value", "name": "密度"},
            "series": [{"type": "line", "smooth": True, "areaStyle": {"opacity": .25}, "data": [[g, round(y, 5)] for g, y in zip(grid, ys)]}]}

def _err_row(i, vals):
    m = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return [i, round(m, 4), round(m - sd, 4), round(m + sd, 4)]


def _viz_errorbar(columns, rows, a):
    groups = {}
    for r in rows:
        xv = r.get(a.x); yv = to_float(r.get(a.y)); zv = r.get(a.z)
        if xv in (None, "") or yv is None:
            continue
        key = (str(xv), str(zv)) if a.z else str(xv)
        groups.setdefault(key, []).append(yv)
    if a.z:
        cats = sorted({k[0] for k in groups})
        series = []; err = []
        for i, cat in enumerate(cats):
            for z in sorted({k[1] for k in groups if k[0] == cat}):
                vals = groups[(cat, z)]
                m = mean(vals)
                series.append({"name": z, "type": "bar", "data": [round(m, 4)]})
                err.append(_err_row(i, vals))
        return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"}, "legend": {"show": True},
                "xAxis": {"type": "category", "data": cats}, "yAxis": {"type": "value", "name": a.y},
                "series": series + [{"type": "errorBar", "data": err, "z": 10}]}
    cats = sorted(groups)
    means = [round(mean(groups[c]), 4) for c in cats]
    err = [_err_row(i, groups[c]) for i, c in enumerate(cats)]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": cats}, "yAxis": {"type": "value", "name": a.y},
            "series": [{"name": a.y, "type": "bar", "data": means},
                       {"type": "errorBar", "data": err, "z": 10}]}


def _viz_mosaic(columns, rows, a):
    agg = {}
    for r in rows:
        xv = r.get(a.x); yv = to_float(r.get(a.y)); zv = r.get(a.z)
        if xv in (None, "") or yv is None or zv in (None, ""):
            continue
        agg[(str(xv), str(zv))] = agg.get((str(xv), str(zv)), 0) + yv
    if not agg:
        return {"title": {"text": "无数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    xs = sorted({k[0] for k in agg}); zs = sorted({k[1] for k in agg})
    totals = {x: sum(agg.get((x, z), 0) for z in zs) for x in xs}
    series = []
    for z in zs:
        series.append({"name": z, "type": "bar", "stack": "pct",
                       "data": [round(100.0 * agg.get((x, z), 0) / totals[x], 2) if totals[x] else 0
                                for x in xs]})
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"}, "legend": {"show": True},
            "xAxis": {"type": "category", "data": xs}, "yAxis": {"type": "value", "name": "占比 %", "max": 100},
            "series": series}


def _viz_km(columns, rows, a):
    ev = []
    for r in rows:
        t = to_float(r.get(a.x)); e = to_float(r.get(a.y))
        if t is None or e is None:
            continue
        ev.append((t, 1 if e >= 0.5 else 0))
    if not ev:
        return {"title": {"text": "无生存数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    ev.sort(key=lambda p: p[0])
    n = len(ev); surv = 1.0; pts = [[0.0, 1.0]]; censored = []
    for i, (t, d) in enumerate(ev):
        at_risk = n - i
        if d == 1:
            surv *= (at_risk - 1) / at_risk
        pts.append([round(t, 4), round(surv, 4)])
        if d == 0:
            censored.append([round(t, 4), round(surv, 4)])
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "value", "name": a.x}, "yAxis": {"type": "value", "name": "生存率", "min": 0, "max": 1},
            "series": [
                {"type": "line", "step": "end", "data": pts, "name": "KM 生存曲线"},
                {"type": "scatter", "data": censored, "symbol": "triangle", "symbolSize": 9, "name": "删失"}
            ]}


def _viz_forest(columns, rows, a):
    studies = []; effects = []; lo = []; hi = []
    for i, r in enumerate(rows):
        s = r.get(a.x) if a.x else "S%d" % (i + 1)
        yv = to_float(r.get(a.y))
        if yv is None:
            continue
        studies.append(str(s)); effects.append(round(yv, 4))
        if a.z:
            ci = to_float(r.get(a.z))
            if ci is not None:
                lo.append(round(yv - ci, 4)); hi.append(round(yv + ci, 4)); continue
        lo.append(None); hi.append(None)
    n = len(studies)
    if n == 0:
        return {"title": {"text": "无数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    err = [[lo[i], i, hi[i]] for i in range(n) if lo[i] is not None]
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": a.y}, "yAxis": {"type": "category", "data": studies},
            "series": [
                {"type": "scatter", "data": [[effects[i], i] for i in range(n)], "symbolSize": 10, "name": "效应量"},
                {"type": "errorBar", "data": err, "z": 5},
                {"type": "line", "data": [[0, i] for i in range(n)], "symbol": "none",
                 "lineStyle": {"type": "dashed", "color": "#999"}, "name": "无效线"}
            ]}


def _viz_roc(columns, rows, a):
    pts = []
    for r in rows:
        s = to_float(r.get(a.x)); l = to_float(r.get(a.y))
        if s is None or l is None:
            continue
        pts.append((s, 1 if l >= 0.5 else 0))
    if len(pts) < 2:
        return {"title": {"text": "数据不足"}, "xAxis": {}, "yAxis": {}, "series": []}
    pts.sort(key=lambda p: -p[0])
    P = sum(1 for _, l in pts if l == 1) or 1
    N = sum(1 for _, l in pts if l == 0) or 1
    curve = [[0.0, 0.0]]; prev_t = 0.0; prev_f = 0.0; auc = 0.0
    for i, (s, l) in enumerate(pts):
        tp = sum(1 for _, ll in pts[:i + 1] if ll == 1)
        fp = sum(1 for _, ll in pts[:i + 1] if ll == 0)
        t = tp / P; f = fp / N
        if l == 1:
            auc += (f - prev_f) * (t + prev_t) / 2.0
        prev_t, prev_f = t, f
        curve.append([round(f, 4), round(t, 4)])
    curve.append([1.0, 1.0])
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"},
            "title": {"text": "ROC（AUC=%.3f）" % round(auc, 3)},
            "xAxis": {"type": "value", "name": "FPR", "min": 0, "max": 1},
            "yAxis": {"type": "value", "name": "TPR", "min": 0, "max": 1},
            "series": [
                {"type": "line", "data": curve, "name": "ROC", "areaStyle": {"opacity": .15}},
                {"type": "line", "data": [[0, 0], [1, 1]], "symbol": "none",
                 "lineStyle": {"type": "dashed", "color": "#999"}, "name": "随机"}
            ]}


def _viz_ridge(columns, rows, a):
    groups = {}
    for r in rows:
        xv = to_float(r.get(a.x)); zv = r.get(a.z)
        if xv is None or (a.z and zv in (None, "")):
            continue
        groups.setdefault(str(zv) if a.z else "all", []).append(xv)
    if not groups:
        return {"title": {"text": "无数值数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    series = []
    for g, vals in groups.items():
        xs = sorted(vals)
        if len(xs) < 2:
            continue
        lo, hi = xs[0], xs[-1]
        grid = [lo + (hi - lo) * i / 99 for i in range(100)]
        if HAS_SCIPY and HAS_NUMPY and len(xs) > 3:
            from scipy.stats import gaussian_kde
            k = gaussian_kde(np.array(xs))
            ys = [float(np.asarray(k(gv)).item()) for gv in grid]
        else:
            nb = 30; w = (hi - lo) / nb or 1
            cnt = [0] * nb
            for v in xs:
                cnt[min(nb - 1, int((v - lo) / w))] += 1
            tot = sum(cnt) * w or 1
            ys = [cnt[min(nb - 1, int((gv - lo) / w))] / tot for gv in grid]
        series.append({"type": "line", "smooth": True, "name": g, "showSymbol": False,
                       "data": [[round(gv, 4), round(y, 5)] for gv, y in zip(grid, ys)]})
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "axis"}, "legend": {"show": True},
            "xAxis": {"type": "value", "name": a.x}, "yAxis": {"type": "value", "name": "密度"},
            "series": series}


def _idw_grid(rows, a, nx=30, ny=30, power=2):
    pts = []
    for r in rows:
        x = to_float(r.get(a.x)); y = to_float(r.get(a.y)); z = to_float(r.get(a.z))
        if None in (x, y, z):
            continue
        pts.append((x, y, z))
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    gx = [min(xs) + (max(xs) - min(xs)) * i / (nx - 1) for i in range(nx)]
    gy = [min(ys) + (max(ys) - min(ys)) * j / (ny - 1) for j in range(ny)]
    grid = []
    for j in range(ny):
        row = []
        for i in range(nx):
            num = 0.0; den = 0.0
            for (px, py, pz) in pts:
                d2 = (px - gx[i]) ** 2 + (py - gy[j]) ** 2
                if d2 == 0:
                    num = pz; den = 1.0; break
                w = 1.0 / (d2 ** (power / 2.0))
                num += w * pz; den += w
            row.append(num / den)
        grid.append(row)
    allv = [v for row in grid for v in row]
    return {"gx": gx, "gy": gy, "grid": grid, "min": min(allv), "max": max(allv)}


def _viz_kriging(columns, rows, a):
    g = _idw_grid(rows, a)
    if not g:
        return {"title": {"text": "插值需要至少 3 个 (x,y,z) 点"}, "xAxis": {}, "yAxis": {}, "series": []}
    data = [[round(g["gx"][i], 4), round(g["gy"][j], 4), round(g["grid"][j][i], 4)]
            for j in range(len(g["gy"])) for i in range(len(g["gx"]))]
    return {"color": VIZ_PALETTE,
            "tooltip": {"position": "top"},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.z, "scale": True},
            "visualMap": {"min": round(g["min"], 4), "max": round(g["max"], 4),
                          "calculable": True, "orient": "horizontal", "left": "center", "bottom": "0%"},
            "series": [{"type": "heatmap", "data": data, "label": {"show": False}}]}


def _marching_squares(grid, gx, gy, level):
    segs = []
    ny = len(grid); nx = len(grid[0])

    def val(i, j):
        return grid[j][i]
    for j in range(ny - 1):
        for i in range(nx - 1):
            v = [val(i, j), val(i + 1, j), val(i + 1, j + 1), val(i, j + 1)]
            x = [gx[i], gx[i + 1], gx[i + 1], gx[i]]
            y = [gy[j], gy[j], gy[j + 1], gy[j + 1]]
            if (v[0] > level) == (v[1] > level) == (v[2] > level) == (v[3] > level):
                continue
            pts = []
            for (p, q) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if (v[p] > level) != (v[q] > level):
                    t = 0.5 if v[p] == v[q] else (level - v[p]) / (v[q] - v[p])
                    pts.append([round(x[p] + (x[q] - x[p]) * t, 4),
                                round(y[p] + (y[q] - y[p]) * t, 4)])
            if len(pts) == 2:
                segs.append(pts)
    return segs


def _segs_to_lines(segs):
    out = []
    for A, B in segs:
        out.append(A); out.append(B); out.append([None, None])
    return out


def _viz_contour(columns, rows, a):
    g = _idw_grid(rows, a)
    if not g:
        return {"title": {"text": "等值线需要至少 3 个 (x,y,z) 点"}, "xAxis": {}, "yAxis": {}, "series": []}
    lo, hi = g["min"], g["max"]
    levels = [lo + (hi - lo) * (k + 1) / 7.0 for k in range(7)]
    series = []
    for li, lv in enumerate(levels):
        segs = _marching_squares(g["grid"], g["gx"], g["gy"], lv)
        series.append({"type": "line", "showSymbol": False, "name": "%.3f" % lv,
                       "data": _segs_to_lines(segs),
                       "lineStyle": {"color": VIZ_PALETTE[li % len(VIZ_PALETTE)]}})
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"}, "legend": {"show": True},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.z, "scale": True},
            "series": series}


def _viz_flow(columns, rows, a):
    data = [[to_float(r.get(a.x)), to_float(r.get(a.y))] for r in rows]
    data = [d for d in data if d[0] is not None and d[1] is not None]
    if not data:
        return {"title": {"text": "无数值数据"}, "xAxis": {}, "yAxis": {}, "series": []}
    mx = median([d[0] for d in data]); my = median([d[1] for d in data])
    return {"color": VIZ_PALETTE, "tooltip": {"trigger": "item"},
            "xAxis": {"type": "value", "name": a.x, "scale": True},
            "yAxis": {"type": "value", "name": a.y, "scale": True},
            "series": [{"type": "scatter", "data": data, "symbolSize": 8,
                        "markLine": {"silent": True, "symbol": "none",
                                     "lineStyle": {"type": "dashed", "color": "#999"},
                                     "data": [{"xAxis": round(mx, 4)}, {"yAxis": round(my, 4)}]}}]}


def _build_viz_option(kind, columns, rows, a):
    fn = {
        "bar": _viz_bar, "hist": _viz_hist, "box": _viz_box, "scatter": _viz_scatter,
        "line": _viz_line, "grouped": _viz_grouped, "stacked": _viz_stacked,
        "pie": _viz_pie, "heatmap": _viz_heatmap, "bubble": _viz_bubble,
        "sankey": _viz_sankey, "network": _viz_network, "volcano": _viz_volcano,
        "density": _viz_density,
        "errorbar": _viz_errorbar, "mosaic": _viz_mosaic, "km": _viz_km, "forest": _viz_forest,
        "roc": _viz_roc, "ridge": _viz_ridge, "kriging": _viz_kriging, "contour": _viz_contour,
        "flow": _viz_flow, "umap": _viz_scatter, "ma": _viz_scatter, "multiline": _viz_line,
    }.get(kind)
    if not fn:
        return None
    opt = fn(columns, rows, a)
    opt.setdefault("title", {"text": a.title or ("%s · %s" % (kind, a.x or ""))})
    opt.setdefault("backgroundColor", "#ffffff" if a.theme == "light" else "#16171c")
    return opt


def _infer_viz_role(kind, role, columns, numeric, exclude=None):
    exclude = set(exclude or [])
    cat_kinds = {"hist", "box", "bar", "density", "ridge", "pie", "mosaic",
                 "grouped", "stacked", "sankey", "network"}
    if role == "x" and kind in cat_kinds:

        for c in columns:
            if c not in numeric and c not in exclude:
                return c
        for c in numeric:
            if c not in exclude:
                return c
        return None
    hints = {
        "x": ["x", "经度", "lon", "longitude", "横", "第一", "seq", "t", "时间", "time",
              "day", "周", "龄", "year", "年", "sample", "样本", "分组", "group", "批"],
        "y": ["y", "纬度", "lat", "latitude", "纵", "值", "value", "表达", "浓度", "强度",
              "荷载", "响应", "score", "指标", "volt", "电压", "电流", "current", "rate",
              "速率", "效应", "effect"],
        "z": ["z", "高程", "elevation", "深度", "depth", "浓度", "值", "value", "温度",
              "temp", "速率", "rate", "压力", "pressure", "组", "group", "类别", "class"],
    }
    for c in numeric:
        if c in exclude:
            continue
        low = c.lower()
        if any(h in low for h in hints.get(role, [])):
            return c
    order = {"y": 0, "z": 1}
    idx = order.get(role, 0)
    avail = [c for c in numeric if c not in exclude]
    if avail:
        return avail[idx] if idx < len(avail) else avail[0]
    return None


def cmd_viz(args):

    disc = None
    if getattr(args, "discipline", None):
        disc = _discipline_index(args.discipline)
        if disc == "auto":
            cols, _ = load_rows(args.input)
            disc = _discipline_index(infer_discipline(cols)["key"])
        elif disc is None:
            die("未知学科: %s（可选: %s / auto）" % (
                args.discipline, ", ".join(d["key"] for d in DISCIPLINE_REGISTRY)))
    if not args.kind:
        if disc:
            args.kind = disc["viz"][0]
        else:
            die("需指定 --kind（图表类型）或 --discipline（自动推断学科并选首图）")
    if args.kind not in VIZ_KINDS:
        die("不支持的图表类型: %s（可选: %s）" % (args.kind, ", ".join(VIZ_KINDS)))
    columns, rows = load_rows(args.input)
    numeric = _numeric_columns(columns, rows)


    assigned = {}
    for role in VIZ_ROLES.get(args.kind, []):
        if not getattr(args, role, None):
            setattr(args, role, _infer_viz_role(args.kind, role, columns, numeric,
                                                 exclude=set(assigned.values())))
        assigned[role] = getattr(args, role)
    miss = [role for role in VIZ_ROLES.get(args.kind, []) if not getattr(args, role, None)]
    if miss:
        die("图表 %s 缺少必需字段: %s（请通过 --x/--y/--z 指定）" % (args.kind, ", ".join(miss)))
    opt = _build_viz_option(args.kind, columns, rows, args)
    if opt is None:
        die("图表 %s 生成失败" % args.kind)
    dark = args.theme == "dark"
    bg, fg = ("#16171c", "#e6e8ef") if dark else ("#ffffff", "#1f2430")
    title = args.title or args.kind
    title_html = html.escape(title, quote=True)
    title_js = json.dumps(title).replace("</", "<\\/")
    opt_js = json.dumps(opt, ensure_ascii=False, default=str).replace("</", "<\\/")
    viz_html = (VIZ_HTML.replace("__TITLE_HTML__", title_html)
            .replace("__TITLE_JS__", title_js)
            .replace("__BG__", bg).replace("__FG__", fg)
            .replace("__THEME__", "dark" if dark else "null")
            .replace("__OPTION__", opt_js))
    out = args.output or ("%s_%s.html" % (os.path.splitext(args.input)[0], args.kind))
    _atomic_write(out, viz_html)
    result = {"kind": args.kind, "output": out,
              "supported_kinds": VIZ_KINDS,
              "numeric_columns": numeric,
              "fields": {"x": args.x, "y": args.y, "z": args.z},
              "note": "自包含 HTML（内嵌 ECharts），浏览器打开即交互图；Ctrl+S 导出 PNG，Ctrl+E 导出 SVG"}
    if disc:
        result["discipline"] = {"key": disc["key"], "name": disc["name"]}
        result["recommended_kinds"] = disc["viz"]
    emit({"status": "ok", "task": "viz", "result": result})

def _t_one_sided_upper(t, df):
    """P(T > t) for Student-t，由双侧 p 推导。"""
    if t is None or df is None or t != t:
        return None
    two = _t_two_sided_p(abs(t), df)
    if two is None:
        return None
    return (two / 2.0) if t >= 0 else (1.0 - two / 2.0)


def _percentile(xs, q):
    """q 为 0-100 百分位，复用 quantile（q 取 0-1）的线性插值。"""
    if not xs:
        return None
    return quantile(sorted(xs), q / 100.0)


def _huber_mean(xs, k=1.345, max_iter=200, tol=1e-9):
    """Huber M-估计稳健均值（零依赖迭代）。"""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    mu = mean(xs)
    s = stdev(xs) or 1.0
    for _ in range(max_iter):
        adj = [min(k * s, max(-k * s, x - mu)) for x in xs]
        new = mu + mean(adj)
        if abs(new - mu) < tol:
            mu = new
            break
        mu = new
    return round(mu, 4)


def _mat_inv(A):
    """高斯-约当求逆（A 为方阵 list[list]）。"""
    n = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max((max(abs(v) for v in row) for row in M), default=0.0)
    if scale == 0.0:
        return None
    thresh = scale * 1e-12
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < thresh:
            return None
        for r in range(n):
            if r == i:
                continue
            f = M[r][i] / M[i][i]
            for c in range(2 * n):
                M[r][c] -= f * M[i][c]
    for i in range(n):
        d = M[i][i]
        for c in range(2 * n):
            M[i][c] /= d
    return [[M[i][n + j] for j in range(n)] for i in range(n)]


def _mat_mul(A, B):
    p = len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(len(B[0]))] for i in range(len(A))]


def _mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def _det_sym(A):
    """对称/一般方阵行列式（高斯消元，零依赖）。奇异或退化返回 None。"""
    n = len(A)
    if n == 0:
        return 0.0
    M = [list(row) for row in A]
    det = 1.0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-12:
            return 0.0
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
            det = -det
        det *= M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n):
                M[r][c] -= f * M[i][c]
    return det


def _eigvalsh_jacobi(M):
    """对称矩阵特征值的 Jacobi 迭代（零依赖），返回降序列表。"""
    n = len(M)
    if n == 0:
        return []
    if n == 1:
        return [float(M[0][0])]
    A = [list(row) for row in M]
    for _ in range(200):
        p, q, maxv = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > abs(maxv):
                    maxv = A[i][j]
                    p, q = i, j
        if abs(maxv) < 1e-12:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        # 旋转角须与更新所用的 R=[[c,s],[-s,c]] 自洽：令 b''=(aqq-app)/2·sin2φ+apq·cos2φ=0
        # ⇒ tan(2φ)=2·apq/(aqq-app)。原分母 (app-aqq) 符号反致 b'' 未消却强行置 0，
        # 得非相似矩阵、特征值失真（仅 a==d 特例正确）。
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for i in range(n):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = c * aip - s * aiq
                A[i][q] = s * aip + c * aiq
                A[p][i] = A[i][p]
                A[q][i] = A[i][q]
        A[p][p] = c * c * app + s * s * aqq - 2.0 * c * s * apq
        A[q][q] = s * s * app + c * c * aqq + 2.0 * c * s * apq
        A[p][q] = 0.0
        A[q][p] = 0.0
    return sorted((A[i][i] for i in range(n)), reverse=True)


def _orthonormal_contrast(k):
    """返回 (k-1) x k 的满秩正交归一对照矩阵（Helmert 归一化）。用于 RM-ANOVA 球形度检验。"""
    m = k - 1
    if m <= 0:
        return []
    C = [[0.0] * k for _ in range(m)]
    for r in range(m):
        # Helmert：第 r 行前 r+1 个为 1，第 r+1 个为 -(r+1)
        for c in range(r + 1):
            C[r][c] = 1.0
        C[r][r + 1] = -(r + 1.0)
        s = math.sqrt(sum(v * v for v in C[r]))
        if s > 0:
            C[r] = [v / s for v in C[r]]
    return C


def _power_emit(payload):
    emit({"status": "ok", "task": "power", "result": payload})


def _posthoc_power_t(d, n1, n2, alpha=None):
    """事后功效（两样本 t，正态近似）。d 为效应量（Cohen's d / Glass's delta / dz），
    n1、n2 为两组样本量；返回功效估计（0-1，四舍五入 4 位）或 None。"""
    a = alpha or _CFG["alpha"]
    if d is None or not n1 or not n2:
        return None
    za = _norm_ppf(1 - a / 2.0)
    if za is None:
        return None
    n = min(n1, n2)
    if n < 2:
        return None
    es = abs(d)
    pw = _norm_cdf(es * math.sqrt(n / 2.0) - za)
    return round(pw, 4) if pw is not None else None


def _posthoc_power_anova(eta2, N, k, alpha=None):
    """事后功效（单因素 ANOVA，正态近似）。eta2 决定 Cohen's f，N 总样本，k 组数。"""
    a = alpha or _CFG["alpha"]
    if eta2 is None or not N or not k or eta2 <= 0 or eta2 >= 1.0 - 1e-12:
        return None
    za = _norm_ppf(1 - a / 2.0)
    if za is None:
        return None
    n_per = N / k
    if n_per < 2:
        return None
    f = math.sqrt(eta2 / (1 - eta2))
    phi = f * math.sqrt(n_per)
    pw = _norm_cdf(phi - za)
    return round(pw, 4) if pw is not None else None


def _low_evidence_advisory_text(reasons):
    """把活检查点原因列表拼接为 advisory 文本；无原因返回 None。"""
    if not reasons:
        return None
    return "；".join(reasons) + "；建议补充样本或改用非参数/稳健方法（见 skill.md 活检查点 §2.7）"


def _tost_p_value(diff, se, df, margin, direction, use_t=True):
    """TOST / 方向性检验核心 p 值（零依赖）。
    direction: 'equivalence'（双向 TOST，差异须落在 [−Δ, +Δ]）/ 'superiority'（单侧 diff>+Δ）/
               'non_inferiority'（单侧 diff>−Δ）。
    use_t=True 用 Student-t（连续均值）；False 用正态近似（比例差）。
    返回 (p, t1, t2)：equivalence 下 t1=(diff+Δ)/se、t2=(diff−Δ)/se；其余 t1 为方向性单侧 t。"""
    if se is None or se <= 0:
        return None, None, None
    up = (lambda t: _t_one_sided_upper(t, df)) if use_t else (lambda t: _norm_sf(t))
    if direction == "equivalence":
        t1 = (diff + margin) / se
        t2 = (diff - margin) / se
        p1 = up(t1)
        p2 = (1.0 - up(t2)) if up(t2) is not None else None
        if p1 is None or p2 is None:
            return None, t1, t2
        return max(p1, p2), t1, t2
    elif direction in ("superiority", "non_inferiority"):
        t = (diff - margin) / se if direction == "superiority" else (diff + margin) / se
        p = up(t)
        return (p, t, None) if p is not None else (None, t, None)
    return None, None, None


def _tost_independent(rows, spec, margin, direction="equivalence", alpha=None):
    a = alpha or _CFG["alpha"]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "tost 需 '因子,数值'"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) != 2:
        return {"error": "需恰好 2 组，当前 %d 组" % len(groups)}
    (na, g1), (nb, g2) = groups.items()
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "每组样本量需 ≥2"}
    s = _two_sample_se(g1, g2)
    se, df, method, diff = s["se"], s["df"], s["method"], s["diff"]
    if se <= 0:
        return {"error": "标准误为 0，无法检验"}
    p, t1, t2 = _tost_p_value(diff, se, df, margin, direction, use_t=True)
    if p is None:
        return {"error": "t 分布计算失败"}
    decision = bool(p < a)
    if direction == "equivalence":
        word = "在等价界内（可认为等效）" if decision else "不能认为等效"
        interp = ("两独立样本 TOST 等价性检验（margin=±%.4f，%s t，α=%g）：均值差 %.4f，"
                  "等价区间 [%.4f, %.4f]，p_TOST=%.4f，%s" % (margin, method, a, diff, -margin, margin, p, word))
        extra = {"t_lower": round(t1, 3), "t_upper": round(t2, 3)}
    else:
        label = "非劣效" if direction == "non_inferiority" else "优效"
        bound = (-margin if direction == "non_inferiority" else margin)
        word = ("可认为%s" % label) if decision else ("不能认为%s" % label)
        interp = ("两独立样本 %s检验（界值 %+.4f，%s t，α=%g）：均值差 %.4f，SE=%.4f，t=%.3f，p=%.4f，%s"
                  % (label, bound, method, a, diff, se, t1, p, word))
        extra = {"t_direction": round(t1, 3)}
    out = {"type": "tost_independent", "direction": direction, "group_a": na, "group_b": nb,
           "mean_diff": round(diff, 4), "margin": margin, "se": round(se, 4), "df": round(df, 2),
           "method": method, "p_tost": _r4(p), "decision": decision, "equivalent": decision,
           "interpretation": interp}
    out.update(extra)
    return out


def _tost_paired(rows, spec, margin, direction="equivalence", alpha=None):
    a = alpha or _CFG["alpha"]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "tost-paired 需 '字段1,字段2'"}
    f1, f2 = parts
    res = _paired_se(rows, f1, f2)
    if res is None:
        return {"error": "有效配对不足（<2）或差值标准差为 0"}
    _, md, _, se, df, n = res
    p, t1, t2 = _tost_p_value(md, se, df, margin, direction, use_t=True)
    if p is None:
        return {"error": "t 分布计算失败"}
    decision = bool(p < a)
    if direction == "equivalence":
        word = "可认为等效" if decision else "不能认为等效"
        interp = ("配对 TOST 等价性检验（margin=±%.4f，α=%g）：平均差 %.4f，等价区间 [%.4f, %.4f]，p_TOST=%.4f，%s"
                  % (margin, a, md, -margin, margin, p, word))
        extra = {"t_lower": round(t1, 3), "t_upper": round(t2, 3)}
    else:
        label = "非劣效" if direction == "non_inferiority" else "优效"
        bound = (-margin if direction == "non_inferiority" else margin)
        word = ("可认为%s" % label) if decision else ("不能认为%s" % label)
        interp = ("配对 %s检验（界值 %+.4f，α=%g）：平均差 %.4f，SE=%.4f，t=%.3f，p=%.4f，%s"
                  % (label, bound, a, md, se, t1, p, word))
        extra = {"t_direction": round(t1, 3)}
    out = {"type": "tost_paired", "direction": direction, "field_a": f1, "field_b": f2, "n_pairs": n,
           "mean_diff": round(md, 4), "margin": margin, "se": round(se, 4), "df": df,
           "p_tost": _r4(p), "decision": decision, "equivalent": decision, "interpretation": interp}
    out.update(extra)
    return out


def _tost_anova(rows, spec, margin, direction="equivalence", alpha=None):
    """单因素 ANOVA 等价性：对所有组对做 TOST（交集-并集原则），全部落在 [−Δ,+Δ] 内方可判定整体等效。
    返回逐对结果 + 整体决策 + 最大绝对均值差（上下文）。"""
    a = alpha or _CFG["alpha"]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "tost --anova 需 '因子,数值'"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) < 2:
        return {"error": "需至少 2 组，当前 %d 组" % len(groups)}
    levels = list(groups.keys())
    means = {lv: mean(groups[lv]) for lv in levels}
    pairs = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lv1, lv2 = levels[i], levels[j]
            g1, g2 = groups[lv1], groups[lv2]
            s = _two_sample_se(g1, g2)
            se, df, diff = s["se"], s["df"], s["diff"]
            if se <= 0:
                pairs.append({"a": lv1, "b": lv2, "error": "标准误为 0"})
                continue
            p, t1, t2 = _tost_p_value(diff, se, df, margin, direction, use_t=True)
            if p is None:
                pairs.append({"a": lv1, "b": lv2, "error": "t 分布计算失败"})
                continue
            pair = {"a": lv1, "b": lv2, "mean_diff": round(diff, 4), "se": round(se, 4),
                    "df": round(df, 2), "p_tost": _r4(p), "equivalent": bool(p < a)}
            if direction == "equivalence":
                pair["t_lower"] = round(t1, 3); pair["t_upper"] = round(t2, 3)
            else:
                pair["t_direction"] = round(t1, 3)
            pairs.append(pair)
    valid = [p_ for p_ in pairs if "p_tost" in p_]
    all_equiv = bool(valid) and all(p_["equivalent"] for p_ in valid)
    max_abs = max(abs(means[lv1] - means[lv2]) for lv1 in levels for lv2 in levels if lv1 != lv2)
    return {"type": "tost_anova", "direction": direction, "factor": fa, "value": fv,
            "n_groups": len(levels), "levels": levels, "margin": margin, "alpha": a,
            "max_abs_mean_diff": round(max_abs, 4),
            "equivalent": all_equiv, "decision": all_equiv,
            "pairs": pairs,
            "interpretation": ("单因素 ANOVA 等价性（全部组对 TOST，margin=±%.4f，α=%g）：最大组间均值差 %.4f，"
                               "%s" % (margin, a, max_abs,
                                       "所有组对均落在等价界内（可认为整体等效）" if all_equiv
                                       else "存在组对超出等价界（不能认为整体等效）"))}


def _tost_proportion(rows, spec, margin, direction="equivalence", alpha=None):
    """两独立样本比例 TOST（正态近似，非合并 SE）。value 列为 0/1 二值。
    检验两比例差 p1−p2 与界值 Δ 的关系（等价/优效/非劣效）。"""
    a = alpha or _CFG["alpha"]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        return {"error": "tost --prop 需 '因子,取值(0/1)'"}
    fa, fv = parts
    groups = _group_values(rows, fa, fv)
    if len(groups) != 2:
        return {"error": "比例 TOST 需恰好 2 组，当前 %d 组" % len(groups)}
    (na, g1), (nb, g2) = groups.items()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return {"error": "每组样本量需 ≥2"}
    x1 = sum(1 for v in g1 if v == 1); x2 = sum(1 for v in g2 if v == 1)
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se <= 0:
        return {"error": "比例差标准误为 0，无法检验"}
    p, t1, t2 = _tost_p_value(diff, se, None, margin, direction, use_t=False)
    if p is None:
        return {"error": "正态近似计算失败"}
    decision = bool(p < a)
    if direction == "equivalence":
        word = "在等价界内（可认为等效）" if decision else "不能认为等效"
        interp = ("两比例 TOST 等价性检验（正态近似，margin=±%.4f，α=%g）：p1=%.4f、p2=%.4f，比例差 %.4f，"
                  "等价区间 [%.4f, %.4f]，p_TOST=%.4f，%s" % (margin, a, p1, p2, diff, -margin, margin, p, word))
        extra = {"z_lower": round(t1, 3), "z_upper": round(t2, 3)}
    else:
        label = "非劣效" if direction == "non_inferiority" else "优效"
        bound = (-margin if direction == "non_inferiority" else margin)
        word = ("可认为%s" % label) if decision else ("不能认为%s" % label)
        interp = ("两比例 %s检验（正态近似，界值 %+.4f，α=%g）：p1=%.4f、p2=%.4f，比例差 %.4f，"
                  "SE=%.4f，z=%.3f，p=%.4f，%s" % (label, bound, a, p1, p2, diff, se, t1, p, word))
        extra = {"z_direction": round(t1, 3)}
    out = {"type": "tost_proportion", "direction": direction, "group_a": na, "group_b": nb,
           "p1": round(p1, 4), "p2": round(p2, 4), "n1": n1, "n2": n2,
           "prop_diff": round(diff, 4), "margin": margin, "se": round(se, 4),
           "p_tost": _r4(p), "decision": decision, "equivalent": decision, "interpretation": interp}
    out.update(extra)
    return out


def cmd_tost(args):
    _apply_cfg(args)
    _, rows = load_rows(args.input)
    margin = to_float(args.margin)
    if margin is None or margin <= 0:
        die("等价界 margin 必须为正数值（>0）")
    direction = (getattr(args, "direction", None) or "equivalence")
    if args.anova:
        res = _tost_anova(rows, args.anova, margin, direction)
    elif args.prop:
        res = _tost_proportion(rows, args.prop, margin, direction)
    elif args.paired:
        res = _tost_paired(rows, args.paired, margin, direction)
    elif args.ttest:
        res = _tost_independent(rows, args.ttest, margin, direction)
    else:
        die("tost 需指定 --ttest / --paired / --anova / --prop 之一")
    emit({"status": "ok", "task": "tost", "result": res})


def _ncf_power(df1, df2, ncp, alpha):
    """非中心 F 的功效（拒绝 H0 概率）。优先 scipy 精确（ncf），零依赖用正态近似。"""
    if ncp is None or ncp < 0:
        ncp = 0.0
    sv = _scipy_try(lambda: float(1.0 - spstats.ncf.cdf(
        float(spstats.f.ppf(1 - alpha, df1, df2)), df1, df2, ncp)))
    if sv is not None:
        return sv
    za = _norm_ppf(1 - alpha)
    if za is None:
        return None
    return _norm_cdf(math.sqrt(ncp) - za)


def cmd_power(args):
    _apply_cfg(args)
    alpha = _CFG["alpha"]
    test = args.test
    power = to_float(args.power) if args.power else None
    n = to_float(args.n)
    za = _norm_ppf(1 - alpha / 2.0 if getattr(args, "alternative", "two-sided") == "two-sided" else 1 - alpha)
    if za is None:
        die("正态分位数计算失败")
    if n is not None and n <= 0:
        die("--n 必须为正样本量")
    if test == "t":
        d = to_float(args.d)
        sd = to_float(args.sd)
        if sd is None:
            sd = 1.0
        elif sd <= 0:
            die("标准差 --sd 必须为正")
        if d is None:
            die("t 检验功效需 --d（均值差）与可选 --sd")
        if d == 0:
            die("效应量 --d 为 0 时无需检验（或所需样本量无穷大）；请提供非零 --d")
        es = d / sd
        if HAS_SCIPY:
            try:
                from scipy.stats import TTestIndPower
                tp = TTestIndPower()
                if power is None and n is not None:
                    pw = tp.power(effect_size=abs(es), nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
                    return _power_emit({"type": "t", "effect_size": round(es, 4), "n_per_group": int(n),
                                        "alpha": alpha, "power": round(float(pw), 4)})
                if n is None and power is not None:
                    nn = tp.solve_power(effect_size=abs(es), power=power, alpha=alpha, ratio=1.0, alternative="two-sided")
                    return _power_emit({"type": "t", "effect_size": round(es, 4),
                                        "required_n_per_group": int(math.ceil(nn)), "alpha": alpha, "power": power})
            except Exception:
                pass
        zb = _norm_ppf(power) if power else None
        if power is None and n is not None:
            pw = _norm_cdf(abs(es) * math.sqrt(n / 2.0) - za)
            return _power_emit({"type": "t", "effect_size": round(es, 4), "n_per_group": int(n),
                                "alpha": alpha, "power": round(pw, 4), "note": "正态近似（零依赖回退）"})
        if n is None and zb is not None:
            nn = 2 * (za + zb) ** 2 / (es ** 2)
            return _power_emit({"type": "t", "effect_size": round(es, 4),
                                "required_n_per_group": int(math.ceil(nn)), "alpha": alpha,
                                "power": power, "note": "正态近似（零依赖回退）"})
        die("功效分析需给定 --power 或 --n 之一（另者求解）")
    if test == "corr":
        r = to_float(args.r)
        if r is None or abs(r) >= 1:
            die("corr 需 |r|<1 的相关系数")
        if n is not None and n < 3:
            die("corr 功效分析需 n>=3（z 变换需 n-3 自由度）")
        zr = 0.5 * math.log((1 + r) / (1 - r))
        if power is None and n is not None:
            pw = _norm_cdf(abs(zr) * math.sqrt(n - 3) - za)
            return _power_emit({"type": "corr", "r": r, "n": int(n), "alpha": alpha, "power": round(pw, 4)})
        if n is None and power is not None:
            zb = _norm_ppf(power)
            nn = int(math.ceil((za + zb) ** 2 / (zr ** 2) + 3))
            return _power_emit({"type": "corr", "r": r, "required_n": nn, "alpha": alpha, "power": power})
        die("功效分析需给定 --power 或 --n 之一")
    if test == "prop":
        p1, p2 = to_float(args.p1), to_float(args.p2)
        if None in (p1, p2) or not (0 < p1 < 1) or not (0 < p2 < 1):
            die("prop 需 0<p1,p2<1")
        delta = abs(p1 - p2)
        if delta == 0:
            die("两组比例需不同")
        if HAS_SCIPY:
            try:
                from scipy.stats import NormalIndPower, proportion_effectsize
                es = proportion_effectsize(p1, p2)
                npw = NormalIndPower()
                if power is None and n is not None:
                    pw = npw.power(effect_size=abs(es), nobs1=n, alpha=alpha, ratio=1.0, alternative="two-sided")
                    return _power_emit({"type": "prop", "p1": p1, "p2": p2, "n_per_group": int(n),
                                        "alpha": alpha, "power": round(float(pw), 4)})
                if n is None and power is not None:
                    nn = npw.solve_power(effect_size=abs(es), power=power, alpha=alpha, ratio=1.0, alternative="two-sided")
                    return _power_emit({"type": "prop", "p1": p1, "p2": p2,
                                        "required_n_per_group": int(math.ceil(nn)), "alpha": alpha, "power": power})
            except Exception:
                pass
        pp_ = (p1 + p2) / 2
        if power is None and n is not None:
            se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
            se0 = math.sqrt(2 * pp_ * (1 - pp_) / n)
            z = (delta / se1) - za * (se0 / se1)
            return _power_emit({"type": "prop", "p1": p1, "p2": p2, "n_per_group": int(n),
                                "alpha": alpha, "power": round(_norm_cdf(z), 4), "note": "正态近似（零依赖回退）"})
        if n is None and power is not None:
            zb = _norm_ppf(power)
            nn = ((za * math.sqrt(2 * pp_ * (1 - pp_)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / delta) ** 2
            return _power_emit({"type": "prop", "p1": p1, "p2": p2,
                                "required_n_per_group": int(math.ceil(nn)), "alpha": alpha,
                                "power": power, "note": "正态近似（零依赖回退）"})
        die("功效分析需给定 --power 或 --n 之一")
    if test == "anova":
        k = int(to_float(args.k)) if args.k else None
        f = to_float(args.f)
        if f is None and args.eta2:
            e2 = to_float(args.eta2)
            f = math.sqrt(e2 / (1 - e2)) if (e2 is not None and 0 < e2 < 1) else None
        if k is None or f is None:
            die("anova 需 --k（组数）与 --f 或 --eta2")
        if HAS_SCIPY:
            try:
                from scipy.stats import FTestAnovaPower
                fp = FTestAnovaPower()
                if power is None and n is not None:
                    pw = fp.power(effect_size=f, nobs=n * k, alpha=alpha, k_groups=k, alternative="two-sided")
                    return _power_emit({"type": "anova", "f": round(f, 4), "k": k, "n_per_group": int(n),
                                        "alpha": alpha, "power": round(float(pw), 4)})
                if n is None and power is not None:
                    nn = fp.solve_power(effect_size=f, power=power, alpha=alpha, k_groups=k)
                    return _power_emit({"type": "anova", "f": round(f, 4), "k": k,
                                        "required_n_per_group": int(math.ceil(nn / k)), "alpha": alpha, "power": power})
            except Exception:
                pass
        if power is None and n is not None:
            phi = f * math.sqrt(n)
            pw = _norm_cdf(phi - za)
            return _power_emit({"type": "anova", "f": round(f, 4), "k": k, "n_per_group": int(n),
                                "alpha": alpha, "power": round(pw, 4), "note": "正态近似（零依赖回退）"})
        if n is None and power is not None:
            zb = _norm_ppf(power)
            nreq = int(math.ceil(((za + zb) / f) ** 2))
            return _power_emit({"type": "anova", "f": round(f, 4), "k": k,
                                "required_n_per_group": nreq, "alpha": alpha, "power": power,
                                "note": "正态近似（零依赖回退）"})
        die("功效分析需给定 --power 或 --n 之一")
    # TOST 等价性检验所需样本量（正态近似，零依赖回退）
    if test == "tost":
        margin = to_float(args.margin)
        sd = to_float(args.sd)
        if sd is None:
            sd = 1.0
        elif sd <= 0:
            die("标准差 --sd 必须为正")
        if margin is None or margin <= 0:
            die("tost 需 --margin（等价界 Δ>0）")
        zb = _norm_ppf(power) if power else None
        if power is None and n is not None:
            if za is None:
                die("正态分位数计算失败")
            zq = math.sqrt(n / 2.0) * margin / sd - za
            pw = _norm_cdf(zq) if zq is not None else None
            return _power_emit({"type": "tost", "margin": margin, "sd": sd,
                                "n_per_group": int(n), "alpha": alpha,
                                "power": round(pw, 4) if pw is not None else None,
                                "note": "正态近似（零依赖回退）"})
        if n is None and zb is not None and za is not None:
            nn = 2 * (za + zb) ** 2 * sd ** 2 / (margin ** 2)
            return _power_emit({"type": "tost", "margin": margin, "sd": sd,
                                "required_n_per_group": int(math.ceil(nn)), "alpha": alpha,
                                "power": power, "note": "正态近似（零依赖回退）"})
        die("tost 需给定 --power 或 --n 之一（另者求解）")
    # 多元回归整体 F 检验功效（检验 R²=0）
    if test == "regression":
        r2 = to_float(args.r2)
        n = to_float(args.n)
        kp = to_float(args.k)
        if r2 is None or kp is None:
            die("regression 需 --r2 与 --k（自变量个数）；计算功效另需 --n，计算所需样本量另需 --power")
        if not (0 < r2 < 1) or kp < 1:
            die("参数非法：需 0<r2<1、k>=1")
        if n is not None and n <= kp + 1:
            die("参数非法：n 需 > k+1")
        f2 = r2 / (1.0 - r2)
        if za is None:
            die("正态分位数计算失败")
        if power is None:
            df1, df2 = kp, n - kp - 1
            ncp = f2 * df2
            pw = _ncf_power(df1, df2, ncp, alpha)
            return _power_emit({"type": "regression", "r2": round(r2, 4), "n": int(n),
                                "k": int(kp), "df1": df1, "df2": round(df2, 2),
                                "cohen_f2": round(f2, 4), "alpha": alpha,
                                "power": round(pw, 4) if pw is not None else None})
        else:
            zb = _norm_ppf(power)
            req_n = int(math.ceil((za + zb) ** 2 / f2 + kp + 1))
            return _power_emit({"type": "regression", "r2": round(r2, 4), "k": int(kp),
                                "cohen_f2": round(f2, 4), "required_n": req_n,
                                "alpha": alpha, "power": power})
    # 生存分析 log-rank 检验样本量（Freedman 近似，等比例两组）
    if test == "survival":
        hr = to_float(args.hr)
        pe = to_float(args.p_event)
        if hr is None or pe is None or hr <= 0 or not (0 < pe < 1):
            die("survival 需 --hr（风险比>0）与 --p_event（事件概率 0-1）")
        if za is None:
            die("正态分位数计算失败")
        if power is None and n is not None:
            E = n * pe
            zb = math.sqrt(max(0.0, E * (math.log(hr) ** 2) * 0.25)) - za
            return _power_emit({"type": "survival", "hr": round(hr, 4), "p_event": round(pe, 4),
                                "n": int(n), "events": round(E, 1), "alpha": alpha,
                                "power": round(_norm_cdf(zb), 4), "note": "log-rank Freedman 近似"})
        if n is None and power is not None:
            zb = _norm_ppf(power)
            E = (za + zb) ** 2 / (math.log(hr) ** 2 * 0.25)
            return _power_emit({"type": "survival", "hr": round(hr, 4), "p_event": round(pe, 4),
                                "required_events": round(E, 1),
                                "required_n": int(math.ceil(E / pe)), "alpha": alpha,
                                "power": power, "note": "log-rank Freedman 近似"})
        die("功效分析需给定 --power 或 --n 之一")
    # 单因素 MANOVA（Pillai 轨迹非中心 F 近似）
    if test == "manova":
        f = to_float(args.f)
        p = to_float(args.p)
        k = to_float(args.k)
        if f is None and args.eta2:
            e2 = to_float(args.eta2)
            f = math.sqrt(e2 / (1.0 - e2)) if (e2 is not None and 0 < e2 < 1) else None
        if f is None or p is None or k is None:
            die("manova 需 --p（因变量个数）、--k（组数）与 --f 或 --eta2")
        if za is None:
            die("正态分位数计算失败")
        if power is None and n is not None:
            df1, df2 = p, n - k - p + 1
            if df2 <= 0:
                die("样本量不足（需 n > k+p-1）")
            ncp = f ** 2 * df2 / (k - 1) if k > 1 else 0.0
            pw = _ncf_power(df1, df2, ncp, alpha)
            return _power_emit({"type": "manova", "f": round(f, 4), "p": int(p), "k": int(k),
                                "n": int(n), "df1": df1, "df2": round(df2, 2), "alpha": alpha,
                                "power": round(pw, 4) if pw is not None else None,
                                "note": "Pillai 轨迹非中心 F 近似"})
        if n is None and power is not None:
            zb = _norm_ppf(power)
            req_n = int(math.ceil((za + zb) ** 2 * (k - 1) / (f ** 2) + k + p - 1))
            return _power_emit({"type": "manova", "f": round(f, 4), "p": int(p), "k": int(k),
                                "required_n": req_n, "alpha": alpha, "power": power,
                                "note": "Pillai 轨迹非中心 F 近似"})
        die("功效分析需给定 --power 或 --n 之一")
    die("未知 test 类型: %s" % test)


def cmd_robust(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    if args.field not in columns:
        die("字段不存在: %s" % args.field)
    xs = _col_floats(rows, args.field, dropna=True)
    if len(xs) < 4:
        die("样本不足（<4）")
    trim = args.trim
    k = int(round(trim * len(xs)))
    sx = sorted(xs)
    trimmed = sx[k:len(sx) - k] if k > 0 else sx
    huber = _huber_mean(xs, args.k)
    out = {"field": args.field, "n": len(xs),
           "arithmetic_mean": round(mean(xs), 4), "median": round(median(xs), 4),
           "std": round(stdev(xs), 4),
           "trimmed_mean": round(mean(trimmed), 4), "trim_proportion": trim,
           "huber_mean": huber, "huber_k": args.k,
           "interpretation": "算术均值 %.4f，%.0f%% 截尾均值 %.4f，Huber(M=%.2f) 稳健均值 %.4f；若稳健估计明显偏离算术均值，提示存在偏态或离群影响" % (
               mean(xs), trim * 100, mean(trimmed), args.k, huber if huber else 0.0)}
    emit({"status": "ok", "task": "robust", "result": out})


def _boot_stat(sample, stat):
    if stat == "mean":
        return mean(sample)
    if stat == "median":
        return median(sample)
    if stat == "sd":
        return stdev(sample)
    return None


def _boot_bca(xs, stats, theta, stat, alpha, seed):
    """BCa（bias-corrected & accelerated）Bootstrap 置信区间，纯 Python 零依赖。
    加速因子 a 由 jackknife 影响值估计；偏差校正 z0 由 Bootstrap 分布中小于点估计的比例估计。
    返回 (lo, hi) 或 None（统计量无变异/样本过小/边界退化时，由调用方回退百分位）。"""
    n = len(xs)
    if n < 3 or theta is None or not stats:
        return None
    # jackknife：大样本子抽样（控制 O(m^2) 复杂度），否则全量
    if n > 3000:
        rng = random.Random(seed)
        base = [xs[i] for i in rng.sample(range(n), 3000)]
    else:
        base = xs
    m = len(base)
    jk = []
    for j in range(m):
        sub = [base[i] for i in range(m) if i != j]
        jk.append(_boot_stat(sub, stat))
    if any(v is None for v in jk):
        return None
    jbar = mean(jk)
    num = 0.0
    den = 0.0
    for v in jk:
        d = jbar - v
        num += d ** 3
        den += d ** 2
    if den <= 0.0:
        return None
    a = num / (6.0 * (den ** 1.5))
    below = sum(1 for s in stats if s < theta)
    p0 = below / len(stats)
    if p0 <= 0.0 or p0 >= 1.0:
        return None
    z0 = _norm_ppf(p0)
    zl = _norm_ppf(alpha / 2.0)
    zh = _norm_ppf(1.0 - alpha / 2.0)
    if z0 is None or zl is None or zh is None:
        return None

    def _adj(z):
        denom = 1.0 - a * (z0 + z)
        if denom <= 0.0:
            return None
        return _norm_cdf(z0 + (z0 + z) / denom)

    al = _adj(zl)
    ah = _adj(zh)
    if al is None or ah is None or not (0.0 < al < 1.0) or not (0.0 < ah < 1.0):
        return None
    sstats = sorted(stats)
    lo = quantile(sstats, al)
    hi = quantile(sstats, ah)
    if lo is None or hi is None:
        return None
    return (lo, hi)


def cmd_bootstrap(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    if args.field not in columns:
        die("字段不存在: %s" % args.field)
    xs = [x for x in _col_floats(rows, args.field, dropna=True) if x is not None]
    if len(xs) < 3:
        die("样本不足（<3）")
    if not (0 < args.alpha < 1):
        die("--alpha 必须介于 0 与 1 之间")
    if args.n <= 0:
        die("重采样次数 --n 必须为正")
    method = getattr(args, "method", "percentile") or "percentile"
    rng = random.Random(args.seed or 0)
    B = args.n
    stats = []
    # numpy 加速：一次性生成重采样索引并对统计量做向量化计算（大 B×n 显著快于逐次纯 Python）
    if HAS_NUMPY and args.stat in ("mean", "median", "sd"):
        try:
            arr = np.asarray(xs, dtype=float)
            n_x = len(xs)
            idx = np.array([rng.randrange(n_x) for _ in range(B * n_x)], dtype=np.intp).reshape(B, n_x)
            if args.stat == "mean":
                stats = arr[idx].mean(axis=1).tolist()
            elif args.stat == "median":
                stats = np.median(arr[idx], axis=1).tolist()
            else:  # sd
                stats = arr[idx].std(axis=1, ddof=1).tolist()
        except Exception:
            stats = []
    if not stats:
        rng2 = random.Random(args.seed or 0)
        for _ in range(B):
            samp = [rng2.choice(xs) for _ in range(len(xs))]
            v = _boot_stat(samp, args.stat)
            if v is not None:
                stats.append(v)
    if not stats:
        die("统计量为空")
    lo_p = _percentile(stats, args.alpha / 2.0 * 100)
    hi_p = _percentile(stats, (1 - args.alpha / 2.0) * 100)
    theta = _boot_stat(xs, args.stat)
    out = {"field": args.field, "statistic": args.stat, "n_resamples": B, "seed": args.seed or 0,
           "method": method, "confidence": round(1 - args.alpha, 4),
           "point_estimate": round(theta, 4) if theta is not None else None,
           "ci_percentile": [round(lo_p, 4), round(hi_p, 4)]}
    if method == "bca":
        bca = _boot_bca(xs, stats, theta, args.stat, args.alpha, args.seed or 0)
        if bca is None:
            out["ci"] = [round(lo_p, 4), round(hi_p, 4)]
            out["ci_bca"] = None
            out["note"] = "BCa 无法估算（统计量无变异或样本过小），已退化为百分位置信区间"
        else:
            out["ci_bca"] = [round(bca[0], 4), round(bca[1], 4)]
            out["ci"] = [round(bca[0], 4), round(bca[1], 4)]
    else:
        out["ci"] = [round(lo_p, 4), round(hi_p, 4)]
    if method == "bca" and out.get("ci_bca"):
        out["interpretation"] = ("对 %s 的 %s 做 %d 次 Bootstrap（seed=%d）：点估计 %.4f，BCa %.1f%% CI = [%.4f, %.4f]"
                                 "（百分位置信区间 [%.4f, %.4f] 供对照）" % (
                                     args.field, args.stat, B, args.seed or 0,
                                     out["point_estimate"], (1 - args.alpha) * 100,
                                     out["ci"][0], out["ci"][1], lo_p, hi_p))
    else:
        out["interpretation"] = "对 %s 的 %s 做 %d 次 Bootstrap 百分位 %.1f%% CI = [%.4f, %.4f]" % (
            args.field, args.stat, B, (1 - args.alpha) * 100, lo_p, hi_p)
    emit({"status": "ok", "task": "bootstrap", "result": out})


_GLM_MAX_ITER = 100


def _glm_irls(X, y, family, max_iter=_GLM_MAX_ITER, tol=1e-8):
    """GLM(IRLS) 纯 Python：logistic / poisson（canonical link）。返回 (beta, cov, mu)。X 已含截距列。"""
    n = len(X); p = len(X[0])
    beta = [0.0] * p
    converged = False
    for _ in range(max_iter):
        eta = [sum(beta[a] * X[i][a] for a in range(p)) for i in range(n)]
        if family == "logistic":
            mu = [_sigmoid(e) for e in eta]
            for i in range(n):
                if mu[i] < 1e-12: mu[i] = 1e-12
                elif mu[i] > 1 - 1e-12: mu[i] = 1 - 1e-12
            gp = [m * (1 - m) for m in mu]
        else:  # poisson
            mu = [max(math.exp(e) if e < 700 else 1e300, 1e-12) for e in eta]
            gp = [max(m, 1e-12) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / gp[i] for i in range(n)]
        XtWX = [[sum(gp[i] * X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        XtWz = [sum(gp[i] * X[i][a] * z[i] for i in range(n)) for a in range(p)]
        bn = gaussian_solve(XtWX, XtWz)
        if bn is None:
            break
        diff = max(abs(bn[a] - beta[a]) for a in range(p))
        beta = bn
        if diff < tol:
            converged = True
            break
    eta = [sum(beta[a] * X[i][a] for a in range(p)) for i in range(n)]
    if family == "logistic":
        mu = [_sigmoid(e) for e in eta]
        for i in range(n):
            if mu[i] < 1e-12: mu[i] = 1e-12
            elif mu[i] > 1 - 1e-12: mu[i] = 1 - 1e-12
        gp = [m * (1 - m) for m in mu]
    else:
        mu = [max(math.exp(e) if e < 700 else 1e300, 1e-12) for e in eta]
        gp = [max(m, 1e-12) for m in mu]
    XtWX = [[sum(gp[i] * X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    cov = _mat_inv(XtWX)
    return beta, cov, mu, converged


def _hc_meat(Xrows, resid, hat, n, p, hc):
    """三明治（异方差一致）协方差的 meat 矩阵。hc: 0/1/2/3。
    HC0: Σ e_i² x_i x_i'；HC1: HC0·n/(n-p)（自由度校正）；
    HC2: Σ e_i²/(1-h_i) x_i x_i'（逐观测杠杆校正）；HC3: Σ e_i²/(1-h_i)² x_i x_i'。"""
    meat = [[0.0] * p for _ in range(p)]
    dof = (n / (n - p)) if n > p else 1.0
    for i in range(n):
        e = resid[i]
        h = hat[i]
        if hc == 0:
            w = e * e
        elif hc == 1:
            w = e * e * dof
        elif hc == 2:
            denom = (1.0 - h)
            w = (e * e / denom) if denom > 1e-12 else e * e
        else:  # hc3
            denom = (1.0 - h)
            w = (e * e / (denom * denom)) if denom > 1e-12 else e * e
        xi = Xrows[i]
        for a in range(p):
            xa = w * xi[a]
            for b in range(p):
                meat[a][b] += xa * xi[b]
    return meat


def _regress_core(Xrows, Y, n, p, hc):
    """回归核心：返回 (beta, resid, XtX_inv, cov_sel, cov_hc3)。
    优先 numpy 向量化（HAS_NUMPY）以加速 O(n·p^2) 的 hat/meat 计算；失败或不可用回退纯 Python，
    行为与历史纯 Python 路径一致。设计矩阵奇异返回 None（由调用方 die）。"""
    if HAS_NUMPY:
        try:
            X = np.asarray(Xrows, dtype=float)
            yv = np.asarray(Y, dtype=float)
            XtX = X.T @ X
            Xty = X.T @ yv
            try:
                XtX_inv = np.linalg.inv(XtX)
            except Exception:
                return None
            try:
                beta = np.linalg.solve(XtX, Xty)
            except Exception:
                return None
            resid = yv - X @ beta
            hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            dof = (n / (n - p)) if n > p else 1.0
            e2 = resid * resid
            w0 = e2
            w1 = e2 * dof
            w2 = np.where((1.0 - hat) > 1e-12, e2 / (1.0 - hat), e2)
            w3 = np.where((1.0 - hat) > 1e-12, e2 / ((1.0 - hat) ** 2), e2)
            wmap = {0: w0, 1: w1, 2: w2, 3: w3}
            wsel = wmap[hc]
            meat_sel = X.T @ (X * wsel[:, None])
            cov_sel = XtX_inv @ meat_sel @ XtX_inv
            if hc == 3:
                cov_hc3 = cov_sel
            else:
                cov_hc3 = XtX_inv @ (X.T @ (X * w3[:, None])) @ XtX_inv
            return (list(map(float, beta)),
                    list(map(float, resid)),
                    [[float(v) for v in row] for row in XtX_inv],
                    [[float(v) for v in row] for row in cov_sel],
                    [[float(v) for v in row] for row in cov_hc3])
        except Exception:
            pass  # 回退纯 Python
    # 纯 Python 回退（行为同历史实现）
    XtX = [[sum(Xrows[i][a] * Xrows[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(Xrows[i][a] * Y[i] for i in range(n)) for a in range(p)]
    beta = gaussian_solve(XtX, Xty)
    if beta is None:
        return None
    pred = [_mat_vec([Xrows[i]], beta)[0] for i in range(n)]
    resid = [Y[i] - pred[i] for i in range(n)]
    XtX_inv = _mat_inv(XtX)
    if XtX_inv is None:
        return None
    hat = []
    for i in range(n):
        xi = Xrows[i]
        tmp = _mat_vec(XtX_inv, xi)
        hat.append(sum(xi[j] * tmp[j] for j in range(p)))
    meat_sel = _hc_meat(Xrows, resid, hat, n, p, hc)
    cov_sel = _mat_mul(_mat_mul(XtX_inv, meat_sel), XtX_inv)
    if hc == 3:
        cov_hc3 = cov_sel
    else:
        cov_hc3 = _mat_mul(_mat_mul(XtX_inv, _hc_meat(Xrows, resid, hat, n, p, 3)), XtX_inv)
    return beta, resid, XtX_inv, cov_sel, cov_hc3


def cmd_regress(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    ycol = args.y
    if ycol not in columns:
        die("因变量字段不存在: %s" % ycol)
    xcols = [c.strip() for c in args.x.split(",") if c.strip()]
    if not xcols:
        die("需至少 1 个自变量（--x）")
    for c in xcols:
        if c not in columns:
            die("自变量字段不存在: %s" % c)
    Y, Xrows = [], []
    for r in rows:
        yv = to_float(r.get(ycol))
        xv = [to_float(r.get(c)) for c in xcols]
        if yv is None or any(v is None for v in xv):
            continue
        Y.append(yv); Xrows.append([1.0] + xv)
    if not Y:
        die("无有效完整观测（因变量或自变量存在缺失/空值）")
    n = len(Y); p = len(Xrows[0])
    if n < p + 2:
        die("样本量不足（需 > %d）" % (p + 1))
    family = (getattr(args, "family", None) or "gaussian").lower()
    names = ["intercept"] + xcols
    if family == "gaussian":
        hc = getattr(args, "hc", None)
        try:
            hc = int(hc)
        except (TypeError, ValueError):
            hc = 3
        if hc not in (0, 1, 2, 3):
            hc = 3
        core = _regress_core(Xrows, Y, n, p, hc)
        if core is None:
            die("设计矩阵奇异，无法估计（存在多重共线性或常量列）")
        beta, resid, XtX_inv, cov_sel, cov_hc3 = core
        s2 = sum(e * e for e in resid) / (n - p)
        # 标准化系数 β* = β_j · (s_xj / s_y)，用于跨自变量比较相对重要性
        y_sd = stdev(Y)
        x_sd = [stdev([Xrows[i][j] for i in range(n)]) for j in range(1, p)]
        coefs = []
        for j in range(p):
            se_robust = math.sqrt(cov_sel[j][j]) if cov_sel[j][j] > 0 else None
            se_hc3 = math.sqrt(cov_hc3[j][j]) if cov_hc3[j][j] > 0 else None
            se_classic = math.sqrt(s2 * XtX_inv[j][j]) if XtX_inv[j][j] > 0 else None
            t = beta[j] / se_robust if se_robust else None
            pval = _t_two_sided_p(abs(t), n - p) if t is not None else None
            std_beta = None
            if j >= 1 and y_sd and y_sd > 0 and x_sd[j - 1] > 0:
                std_beta = beta[j] * (x_sd[j - 1] / y_sd)
            coefs.append({"term": names[j], "coef": round(beta[j], 4),
                          "std_beta": round(std_beta, 4) if std_beta is not None else None,
                          "se_classic": round(se_classic, 4) if se_classic else None,
                          "se_robust": round(se_robust, 4) if se_robust else None,
                          "hc_type": hc,
                          "t_robust": round(t, 3) if t is not None else None,
                          "p_robust": _r4(pval) if pval is not None else None,
                          "se_hc3": round(se_hc3, 4) if se_hc3 else None,
                          "t_hc3": round(beta[j] / se_hc3, 3) if se_hc3 else None,
                          "p_hc3": _r4(_t_two_sided_p(abs(beta[j] / se_hc3), n - p)) if se_hc3 else None,
                          "significant": _sig(pval)})
        ss_res = sum(e * e for e in resid)
        mY = mean(Y)
        ss_tot = sum((y - mY) ** 2 for y in Y)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
        # E3 杠杆诊断：hat_i = x_i·(XtX)^-1·x_i，阈值 2p/n（Belsley 等惯例）
        lev_thr = 2.0 * p / n
        hats = []
        for xi in Xrows:
            tmp = _mat_vec(XtX_inv, xi)
            hats.append(sum(xi[j] * tmp[j] for j in range(p)))
        n_hi = sum(1 for h in hats if h > lev_thr)
        max_hat = max(hats) if hats else None
        out = {"terms": coefs, "n": n, "df_resid": n - p, "hc_type": hc,
               "r_squared": round(r2, 4) if r2 is not None else None,
               "diagnostics": {"leverage_threshold": round(lev_thr, 4),
                               "max_leverage": round(max_hat, 4) if max_hat is not None else None,
                               "n_high_leverage": n_hi},
               "note": "标准误给出经典 OLS 与可选 HC（HC0/1/2/3，默认 HC3）异方差稳健估计；异方差存在时以 se_robust 为准；std_beta 为标准化系数（β·s_x/s_y），可跨自变量比较重要性",
               "interpretation": "R²=%.3f；HC%d 稳健 t 检验见 terms（se_robust 为准），std_beta 示标准化效应" % (r2 if r2 is not None else 0.0, hc)}
        if n_hi:
            _warn("回归杠杆诊断：%d 个观测 hat 值 > 2p/n=%.3f（最大 %.3f），估计可能受高杠杆点主导，建议复核或稳健处理"
                  % (n_hi, lev_thr, max_hat))
        emit({"status": "ok", "task": "regress", "result": out}); return
    elif family in ("logistic", "poisson"):
        if family == "logistic":
            uniq = sorted(set(Y))
            if len(uniq) != 2:
                die("logistic 要求二分类因变量（恰 2 个水平）")
            m1 = uniq[1]
            yvec = [1.0 if y == m1 else 0.0 for y in Y]
        else:
            yvec = [float(y) for y in Y]
        beta, cov, mu, converged = _glm_irls(Xrows, yvec, family)
        elabel = "odds_ratio" if family == "logistic" else "incidence_rate_ratio"

        def _ex(x):
            # E3：完全分离/发散时 |beta|>700 会使 math.exp 溢出——上界返回 None（不可表示），下界归 0
            if x is None or x != x:
                return None
            if x > 700.0:
                return None
            if x < -700.0:
                return 0.0
            return math.exp(x)
        coefs = []
        for j in range(p):
            se = math.sqrt(cov[j][j]) if cov and cov[j][j] > 0 else None
            z = beta[j] / se if se else None
            pval = _norm_sf(abs(z)) * 2.0 if z is not None else None
            ratio = _ex(beta[j])
            lo = _ex(beta[j] - 1.96 * se) if se else None
            hi = _ex(beta[j] + 1.96 * se) if se else None
            ci = [round(lo, 4), round(hi, 4)] if (lo is not None and hi is not None) else None
            coefs.append({"term": names[j], "coef": round(beta[j], 4),
                          "se": round(se, 4) if se else None,
                          "z": round(z, 3) if z is not None else None,
                          "p_value": _r4(pval) if pval is not None else None,
                          "significant": _sig(pval), elabel: round(ratio, 4) if ratio is not None else None,
                          "ci_95_%s" % elabel: ci})
        out = {"family": family, "terms": coefs, "n": n, "converged": converged,
               "note": "GLM(%s) 经 IRLS 估计；%s 与 95%% CI 已给出（零依赖回退）" % (family, elabel),
               "interpretation": "%s 回归（IRLS 极大似然）：各预测因子的 %s 与显著性见 terms。" % (family, elabel)}
        if family == "logistic" and mu:
            pmin, pmax = min(mu), max(mu)
            out["diagnostics"] = {"fitted_prob_min": round(pmin, 6),
                                  "fitted_prob_max": round(pmax, 6)}
            if pmin < 1e-6 or pmax > 1.0 - 1e-6:
                _warn("logistic 拟合概率出现极端值（min=%.2g, max=%.6f）：疑似完全分离，"
                      "系数与标准误可能发散，建议复核预测变量或做 Firth/惩罚回归" % (pmin, pmax))
        if not converged:
            out["note"] += "；警告：IRLS 未在 %d 次迭代内收敛（可能完全分离/共线性），系数标准误不可靠" % _GLM_MAX_ITER
        emit({"status": "ok", "task": "regress", "result": out}); return
    else:
        die("不支持的 family: %s（可选 gaussian/logistic/poisson）" % family)


def cmd_sensitivity(args):
    _apply_cfg(args)
    _, rows = load_rows(args.input)
    if args.ttest:
        spec = args.ttest
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 2:
            die("sensitivity --ttest 需 '因子,数值'")
        full = _ttest_independent(rows, spec)
        if "error" in full:
            emit({"status": "error", "task": "sensitivity", "message": full["error"]}); return
        trimmed = _remove_l3_outliers(rows, parts[0], parts[1])
        trim = _ttest_independent(trimmed, spec)
        n_full = full.get("group_a", {}).get("n", 0) + full.get("group_b", {}).get("n", 0)
        n_trim = trim.get("group_a", {}).get("n", 0) + trim.get("group_b", {}).get("n", 0)
        n_removed = n_full - n_trim
        fp, tp = full.get("p_value"), trim.get("p_value")
        stable = _sig(fp) == _sig(tp)
        out = {"test": "独立样本 t 检验", "spec": spec, "n_removed_outliers": n_removed,
               "full": {"n": n_full, "p_value": _r4(fp), "significant": _sig(fp)},
               "after_outlier_removal": {"n": n_trim, "p_value": _r4(tp), "significant": _sig(tp)},
               "consistency": "稳定（结论未因离群点改变）" if stable else "敏感（剔除离群点后结论改变）",
               "interpretation": "采用 IQR×3.0（L3）剔除离群点后共移除 %d 条，t 检验结论%s。" % (
                   n_removed, "保持稳定" if stable else "发生改变，提示原结论对离群点敏感，建议改用稳健方法（robust/regress HC3）")}
        emit({"status": "ok", "task": "sensitivity", "result": out})
    elif args.anova:
        spec = args.anova
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 2:
            die("sensitivity --anova 需 '因子,数值'（单因素）")
        gcol, vcol = parts[0], parts[1]
        full_pairs = [(str(r.get(gcol)), to_float(r.get(vcol))) for r in rows if to_float(r.get(vcol)) is not None]
        full = _one_way_anova(full_pairs)
        trimmed = _remove_l3_outliers(rows, gcol, vcol)
        trim_pairs = [(str(r.get(gcol)), to_float(r.get(vcol))) for r in trimmed if to_float(r.get(vcol)) is not None]
        trim = _one_way_anova(trim_pairs)
        fp, tp = full.get("p_value"), trim.get("p_value")
        stable = _sig(fp) == _sig(tp)
        out = {"test": "单因素 ANOVA", "spec": spec, "n_removed_outliers": len(rows) - len(trimmed),
               "full": {"n": full.get("n"), "p_value": _r4(fp), "significant": _sig(fp)},
               "after_outlier_removal": {"n": trim.get("n"), "p_value": _r4(tp), "significant": _sig(tp)},
               "consistency": "稳定（结论未因离群点改变）" if stable else "敏感（剔除离群点后结论改变）",
               "interpretation": "采用 IQR×3.0（L3）剔除离群点后共移除 %d 条，ANOVA 结论%s。" % (
                   len(rows) - len(trimmed), "保持稳定" if stable else "发生改变，提示原结论对离群点敏感")}
        emit({"status": "ok", "task": "sensitivity", "result": out})
    else:
        die("sensitivity 需 --ttest 因子,数值 或 --anova 因子,数值")


def _reml_tau2(es, se, k):
    """REML 估计 τ²：IGLS 迭代 τ² = [Σw'(yᵢ-μ)² - (k-1)]/Σw'，其中 w'=1/(se_i²+τ²)，μ 为随机效应加权均值。
    零依赖；迭代至收敛或 τ² 触底为 0。"""
    tau2 = 0.01
    for _ in range(200):
        w = [1.0 / (se[i] ** 2 + tau2) for i in range(k)]
        sw = sum(w)
        mu = sum(w[i] * es[i] for i in range(k)) / sw
        num = sum(w[i] * (es[i] - mu) ** 2 for i in range(k)) - (k - 1)
        new = num / sw if sw > 0 else 0.0
        if new < 0:
            new = 0.0
        if abs(new - tau2) < 1e-9:
            tau2 = new
            break
        tau2 = new
    return max(0.0, tau2)


def _egger_test(es, se):
    """Egger 回归检验出版偏倚：Y=es/se 对 X=1/se 的加权回归（权重 1/se²），截距≠0 提示不对称。零依赖。"""
    k = len(es)
    if k < 3 or any(s <= 0 for s in se):
        return None
    X = [1.0 / s for s in se]
    Y = [e / s for e, s in zip(es, se)]
    W = [1.0 / (s * s) for s in se]
    Sw = sum(W); Sx = sum(W[i] * X[i] for i in range(k))
    Sy = sum(W[i] * Y[i] for i in range(k))
    Sxx = sum(W[i] * X[i] * X[i] for i in range(k))
    Sxy = sum(W[i] * X[i] * Y[i] for i in range(k))
    Minv = _mat_inv([[Sw, Sx], [Sx, Sxx]])
    if Minv is None:
        return None
    denom = Sw * Sxx - Sx * Sx
    if denom == 0:
        return None
    b1 = (Sw * Sxy - Sx * Sy) / denom
    a = (Sy - b1 * Sx) / Sw
    var_a = Minv[0][0]
    if var_a <= 0:
        return None
    z = a / math.sqrt(var_a)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    sig = _sig(p)
    return {"intercept": round(a, 4), "intercept_se": round(math.sqrt(var_a), 4),
            "z": round(z, 3), "p_value": _r4(p), "significant": sig,
            "interpretation": "Egger 回归截距%s（z=%.3f, p=%s），提示存在小样本/出版偏倚所致的不对称" % (
                "显著" if sig else "不显著", z, _fmt_p(p))}


def _meta_pool(studies, alpha=None, estimator="dl"):
    """元分析核心：固定/随机效应合并、τ²（DL/REML/HS）、I²/H²、Egger 出版偏倚、漏斗图数据、
    Rosenthal 失安全 N、随机效应预测区间、森林图数据。studies: [{'label','es','se'}...]。零依赖。"""
    a = alpha or _CFG["alpha"]
    zc = _norm_ppf(0.5 + (1 - a) / 2.0) or 1.959964
    k = len(studies)
    if k == 0:
        return {"error": "无有效研究"}
    es = [s["es"] for s in studies]
    warnings = []
    se = []
    for s in studies:
        if s["se"] and s["se"] > 0:
            se.append(s["se"])
        else:
            warnings.append("研究 %r 的标准误 se 非正（%r），已按极小值 1e-9 处理，该研究的合并权重极不可靠"
                           % (s.get("label"), s["se"]))
            se.append(1e-9)
    w = [1.0 / (s * s) for s in se]
    sw = sum(w)
    pooled_fe = sum(wi * ei for wi, ei in zip(w, es)) / sw
    se_fe = 1.0 / math.sqrt(sw)
    Q = sum(wi * (ei - pooled_fe) ** 2 for wi, ei in zip(w, es))
    df = k - 1
    Qp = _chi2_sf(Q, df)
    denom = sw - sum(wi * wi for wi in w) / sw
    tau2_dl = max(0.0, (Q - df) / denom) if (df > 0 and denom > 0) else 0.0
    if estimator == "fixed":
        tau2 = 0.0
    elif estimator == "hs":
        tau2 = max(0.0, (sum(wi * (ei - pooled_fe) ** 2 for wi, ei in zip(w, es)) - df) / sw)
    elif estimator == "reml":
        tau2 = _reml_tau2(es, se, k)
    else:  # dl (默认)
        tau2 = tau2_dl
    wre = [1.0 / (se[i] ** 2 + tau2) for i in range(k)]
    swre = sum(wre)
    pooled_re = sum(wre[i] * es[i] for i in range(k)) / swre if swre > 0 else pooled_fe
    se_re = 1.0 / math.sqrt(swre) if swre > 0 else se_fe
    i2 = max(0.0, (Q - df) / Q) * 100.0 if (Q > 0 and df > 0) else 0.0
    h2 = Q / df if df > 0 else 1.0

    def ci(p, s):
        if p is None or s is None:
            return None
        return [round(p - zc * s, 4), round(p + zc * s, 4)]

    forest = []
    for i in range(k):
        lo, hi = es[i] - zc * se[i], es[i] + zc * se[i]
        forest.append({"study": studies[i]["label"], "es": round(es[i], 4), "se": round(se[i], 4),
                       "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                       "weight_fixed": round(w[i] / sw, 4),
                       "weight_random": round(wre[i] / swre, 4) if swre > 0 else None})
    funnel = [{"study": studies[i]["label"], "es": round(es[i], 4), "se": round(se[i], 4)} for i in range(k)]
    egger = _egger_test(es, se)
    fsn = max(0, int(round((Q - df) / (zc ** 2)))) if Q > df else 0
    pred_int = None
    if k - 2 > 0 and tau2 > 0:
        tcrit = _t_ppf(1.0 - a / 2.0, k - 2)
        if tcrit is not None:
            sp = math.sqrt(tau2 + se_re ** 2)
            pred_int = [round(pooled_re - tcrit * sp, 4), round(pooled_re + tcrit * sp, 4)]
    pub = {"egger": egger, "fail_safe_n": fsn}
    if egger and egger.get("significant"):
        pub["fail_safe_note"] = "需额外 %d 项阴性研究才能使合并 p>%s（数值越大越稳健）" % (fsn, a)
    return {
        "k": k, "measure": studies[0].get("measure", "effect_size"),
        "estimator": estimator,
        "pooled_fixed": {"es": round(pooled_fe, 4), "se": round(se_fe, 4), "ci": ci(pooled_fe, se_fe)},
        "pooled_random": {"es": round(pooled_re, 4), "se": round(se_re, 4), "ci": ci(pooled_re, se_re)},
        "heterogeneity": {"Q": round(Q, 4), "df": df, "Q_p_value": _r4(Qp),
                          "tau2": round(tau2, 4), "tau2_method": estimator, "tau2_dl": round(tau2_dl, 4),
                          "I2_percent": round(i2, 2), "H2": round(h2, 4)},
        "model_recommendation": "随机效应模型（I²≥50%，存在实质异质性）" if i2 >= 50
        else "固定效应模型（I²<50%，异质性低）",
        "publication_bias": pub,
        "prediction_interval": pred_int,
        "funnel": funnel,
        "forest": forest,
        "warnings": warnings if warnings else None,
    }


def _meta_forest_html(meta, title="森林图"):
    """生成零依赖、自包含的森林图 HTML（内联 CSS 画点 + 置信区间条，无需 JS）。"""
    studies = list(meta["forest"])
    fe = meta["pooled_fixed"]; re = meta["pooled_random"]
    studies.append({"study": "合并(固定效应)", "es": fe["es"], "ci_low": fe["ci"][0], "ci_high": fe["ci"][1], "pooled": True})
    studies.append({"study": "合并(随机效应)", "es": re["es"], "ci_low": re["ci"][0], "ci_high": re["ci"][1], "pooled": True})
    allv = [s["es"] for s in studies] + [s["ci_low"] for s in studies] + [s["ci_high"] for s in studies]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    rows = ""
    for s in studies:
        left = (s["ci_low"] - lo) / span * 100.0
        width = (s["ci_high"] - s["ci_low"]) / span * 100.0
        dot = (s["es"] - lo) / span * 100.0
        rows += ("<tr class='%s'><td class='lab'>%s</td>"
                 "<td class='bar'><div class='track'><div class='ci' style='left:%.2f%%;width:%.2f%%'></div>"
                 "<div class='pt' style='left:%.2f%%'></div></div></td>"
                 "<td class='val'>%.3f [%.3f, %.3f]</td></tr>" % (
                     "pool" if s.get("pooled") else "", s["study"], left, width, dot,
                     s["es"], s["ci_low"], s["ci_high"]))
    return ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>%s</title><style>"
            "body{font-family:sans-serif;margin:24px;color:#222}"
            "h2{font-size:18px}.lab{width:160px;font-size:13px;padding:2px 6px;text-align:right}"
            ".bar{width:60%%}.track{position:relative;height:14px}"
            ".ci{position:absolute;top:5px;height:4px;background:#555}"
            ".pt{position:absolute;top:0;width:0;height:14px;border-left:3px solid %s}"
            "tr.pool .ci{border-left:2px solid #c0392b;background:#c0392b}"
            ".val{font-size:12px;padding-left:10px;color:#444}"
            ".scale{font-size:11px;color:#999;padding-left:166px}</style></head><body>"
            "<h2>%s</h2><table>%s</table>"
            "<div class='scale'>%.3f ——— %.3f</div></body></html>" % (
                title, "#c0392b", title, rows, lo, hi))


def cmd_meta(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    mode = args.mode
    spec = [p.strip() for p in args.spec.split(",")] if args.spec else None
    studies = []
    measure = "effect_size"
    if mode == "es":
        if not spec:
            if "ci_low" in columns and "ci_high" in columns:
                spec = ["study", "es", "ci_low", "ci_high"]
            else:
                spec = ["study", "es", "se"]
        s_i, e_i = spec[0], spec[1]
        if len(spec) >= 4:
            cl_i, ch_i = spec[2], spec[3]
            for r in rows:
                lab = r.get(s_i); e = to_float(r.get(e_i))
                cl, ch = to_float(r.get(cl_i)), to_float(r.get(ch_i))
                if lab is None or e is None or cl is None or ch is None:
                    continue
                zc = _norm_ppf(0.975) or 1.959964
                s = (ch - cl) / (2 * zc)
                if s <= 0:
                    continue
                studies.append({"label": str(lab), "es": e, "se": s, "measure": measure})
        else:
            se_i = spec[2] if len(spec) > 2 else None
            for r in rows:
                lab = r.get(s_i); e = to_float(r.get(e_i))
                if lab is None or e is None:
                    continue
                s = to_float(r.get(se_i)) if se_i else None
                if s is None or s <= 0:
                    continue
                studies.append({"label": str(lab), "es": e, "se": s, "measure": measure})
    elif mode == "cont":
        if not spec:
            spec = ["study", "n1", "n2", "m1", "m2", "sd1", "sd2"]
        s_i, n1_i, n2_i, m1_i, m2_i, sd1_i, sd2_i = spec[:7]
        measure = "cohens_d"
        for r in rows:
            lab = r.get(s_i)
            n1, n2 = to_float(r.get(n1_i)), to_float(r.get(n2_i))
            m1, m2 = to_float(r.get(m1_i)), to_float(r.get(m2_i))
            sd1, sd2 = to_float(r.get(sd1_i)), to_float(r.get(sd2_i))
            if None in (lab, n1, n2, m1, m2, sd1, sd2) or n1 < 2 or n2 < 2:
                continue
            sp = math.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
            if sp <= 0:
                continue
            d = (m2 - m1) / sp
            se_d = math.sqrt(1.0 / n1 + 1.0 / n2 + d ** 2 / (2 * (n1 + n2)))
            # Hedges g 小样本校正
            g = d * (1.0 - 3.0 / (4 * (n1 + n2 - 2) - 1)) if (n1 + n2 - 2) > 1 else d
            studies.append({"label": str(lab), "es": d, "se": se_d, "measure": "cohens_d", "hedges_g": round(g, 4)})
    elif mode in ("or2x2", "rr2x2"):
        if not spec:
            spec = ["study", "a", "b", "c", "d"]
        s_i, a_i, b_i, c_i, d_i = spec[:5]
        measure = "logOR" if mode == "or2x2" else "logRR"
        for r in rows:
            lab = r.get(s_i)
            a, b, c, d = (to_float(r.get(x)) for x in (a_i, b_i, c_i, d_i))
            if None in (lab, a, b, c, d) or a <= 0 or b <= 0 or c <= 0 or d <= 0:
                continue
            if mode == "or2x2":
                es = math.log((a * d) / (b * c))
                se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            else:
                p1, p2 = a / (a + b), c / (c + d)
                es = math.log(p1 / p2) if p1 > 0 and p2 > 0 else None
                se = math.sqrt(1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d)) if es is not None else None
            if es is None or se is None:
                continue
            studies.append({"label": str(lab), "es": es, "se": se, "measure": measure})
    else:
        die("未知 meta --mode: %s（可选 es/cont/or2x2/rr2x2）" % mode)
    if not studies:
        die("未解析到任何有效研究（请检查 --mode 与列名；cont 需 n1,n2,m1,m2,sd1,sd2；2x2 需 a,b,c,d）")
    meta = _meta_pool(studies, args.alpha, getattr(args, "estimator", None) or "dl")
    if "error" in meta:
        die(meta["error"])
    if args.output:
        if args.output.lower().endswith(".html"):
            _atomic_write(args.output, _meta_forest_html(meta, title="元分析森林图（%s）" % mode), encoding="utf-8")
            meta["output"] = args.output
        else:
            _atomic_write(args.output, json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            meta["output"] = args.output
    emit({"status": "ok", "task": "meta", "result": meta})


def _sigmoid(z):
    if z < -30:
        return 0.0
    if z > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))




# ======================================================================
#  T1④ 统一检验标准块 + T2⑤/T3⑤/T1①/T1②/T1③/T2⑦ 扩展能力
#  全部纯 Python 实现，numpy/scipy 仅在存在时加速并优雅回退。
# ======================================================================

def _f_ppf(q, df1, df2):
    """F 分布分位数（逆 CDF）。零依赖用 _f_sf 二分反解。"""
    if q is None or q <= 0 or q >= 1 or df1 <= 0 or df2 <= 0:
        return None
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _f_sf(mid, df1, df2) > 1.0 - q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _q_ppf(alpha, k, df):
    """学生化极差分布临界值 q(α;k,df)。优先 scipy，否则正态近似。"""
    sv = _scipy_try(lambda: float(__import__("scipy").stats.studentized_range.ppf(1.0 - alpha, k, df)))
    if sv is not None:
        return sv
    za = _norm_ppf(1.0 - alpha / 2.0)
    return za * math.sqrt(2.0)


def _test_block(stat_name, stat, p, df1=None, df2=None, df=None, effect=None,
                effect_kind=None, ci=None, bf10=None, power=None, n=None):
    """统一检验标准块：把各主检验的核心量打包成一致结构（stat/p/df/效应量/CI/BF10/功效）。"""
    blk = {"test": stat_name, "statistic": _r4(stat), "p_value": _r4(p),
           "significant": _sig(p)}
    if df is not None:
        blk["df"] = _r4(df)
    else:
        if df1 is not None:
            blk["df1"] = _r4(df1)
        if df2 is not None:
            blk["df2"] = _r4(df2)
    if effect is not None:
        blk["effect_size"] = {"value": _r4(effect), "kind": effect_kind}
        lbl = _effect_label(effect_kind, effect) if effect_kind else None
        if lbl:
            blk["effect_size"]["label"] = lbl
    if ci is not None:
        blk["effect_ci_95"] = ci
    if bf10 is not None:
        blk["bayes_factor_10"] = _r4(bf10)
    if power is not None:
        blk["posthoc_power"] = _r4(power)
    if n is not None:
        blk["n"] = n
    return blk


def _cramers_v(contingency):
    """列联表 Cramer's V（零依赖）。contingency: list[list[int]]。"""
    try:
        k = len(contingency)
        if k < 2:
            return None
        col_n = len(contingency[0])
        n = sum(sum(r) for r in contingency)
        if n == 0:
            return None
        chi2 = 0.0
        row_t = [sum(r) for r in contingency]
        col_t = [sum(contingency[i][j] for i in range(k)) for j in range(col_n)]
        for i in range(k):
            for j in range(col_n):
                exp = row_t[i] * col_t[j] / n
                if exp > 0:
                    chi2 += (contingency[i][j] - exp) ** 2 / exp
        v = math.sqrt(chi2 / (n * (min(k, col_n) - 1)))
        return round(v, 4)
    except Exception:
        return None


def _eta_assoc(cat_vals, num_vals):
    """分类→连续 的关联强度 η² = SS_between / SS_total。"""
    d = OrderedDict()
    for c, x in zip(cat_vals, num_vals):
        if x is not None:
            d.setdefault(str(c), []).append(x)
    if len(d) < 2:
        return None
    allv = [x for g in d.values() for x in g]
    grand = mean(allv)
    sst = sum((x - grand) ** 2 for x in allv)
    ssb = sum(len(g) * (mean(g) - grand) ** 2 for g in d.values())
    return round(ssb / sst, 4) if sst > 0 else None


def _smd(a, b):
    """两样本标准化均值差（Cohen's d；合并标准差）。"""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = mean(a), mean(b)
    va, vb = stdev(a) ** 2, stdev(b) ** 2
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return round((ma - mb) / sp, 4) if sp > 0 else None


def _compact_letters(labels, means, sig):
    """字母标记法（compact letter display）：按均值排序，显著不同组必有不相交字母。
    sig[(i,j)]=True 表示 i,j 差异显著。返回 {label: letter}。"""
    n = len(labels)
    order = sorted(range(n), key=lambda i: means[i] if means[i] is not None else 0.0)
    letters = ["" for _ in range(n)]
    for step, i in enumerate(order):
        used_by_different = set()
        used_by_same = set()
        for q in order[:step]:
            if sig.get((min(q, i), max(q, i)), True):
                used_by_different |= set(letters[q])
            else:
                used_by_same |= set(letters[q])
        chosen = None
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch in used_by_same:
                chosen = ch
                break
            if ch not in used_by_different:
                chosen = ch
                break
        letters[i] = chosen or "a"
    return {labels[i]: letters[i] for i in range(n)}


def _dunnett(groups, control, alpha):
    """Dunnett 多重比较（多对一，control vs 各组）。返回两两结果 + CLD。"""
    a = alpha or _CFG["alpha"]
    labels = list(groups.keys())
    if control not in labels or len(labels) < 2:
        return {"error": "Dunnett 需指定有效的对照水平(--control)"}
    cg = groups[control]
    mc = mean(cg); nc = len(cg); vc = stdev(cg) ** 2 if len(cg) > 1 else 0.0
    k = len(labels)
    out = []
    sig = {}
    for lab in labels:
        if lab == control:
            continue
        g = groups[lab]
        ng = len(g); mg = mean(g); vg = stdev(g) ** 2 if len(g) > 1 else 0.0
        se = math.sqrt(vc / nc + vg / ng) if (nc > 0 and ng > 0) else None
        if se is None or se <= 0:
            out.append({"compared": "%s vs %s" % (lab, control), "t": None, "p_value": None})
            continue
        t = (mg - mc) / se
        df = nc + ng - 2
        # Dunnett 临界 = q(k,df)/sqrt(2)；无 scipy 时用 Bonferroni t 临界（保守）
        qc = _q_ppf(a, k, df)
        if qc is not None and qc > 0 and abs(qc - _norm_ppf(1 - a / 2.0) * math.sqrt(2.0)) > 1e-6:
            crit = qc / math.sqrt(2.0)
        else:
            crit = _t_ppf(1 - a / (2 * (k - 1)), df) or _t_ppf(1 - a / 2.0, df)
        p = _t_two_sided_p(abs(t), df)
        sig[(control, lab)] = abs(t) > crit
        blk = _test_block("Dunnett-t", t, p, df2=df, effect=mg - mc)
        blk["mean_diff"] = round(mg - mc, 4)
        blk["critical_t"] = round(crit, 4)
        blk["significant"] = bool(abs(t) > crit)  # Dunnett 用临界值判定，覆盖统一块默认的 p<a
        out.append(blk)
    means = {l: (mean(g) if g else None) for l, g in groups.items()}
    cld = _compact_letters(labels, [means[l] for l in labels],
                           _sig_map(sig, labels))
    return {"method": "Dunnett (many-to-one)", "control": control, "alpha": a,
            "comparisons": out, "compact_letters": cld}


def _sig_map(sig, labels):
    """把 Dunnett 的 control-vs-x 显著关系扩散为全两两 sig 字典（用于 CLD）。"""
    m = {min(a, b): max(a, b) for (a, b) in sig}
    d = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            if (li in m and lj in m) or (lj in m and li in m):
                d[(li, lj)] = m.get(li, m.get(lj, True))
            else:
                d[(li, lj)] = True
    return d


def _nemenyi(groups, alpha):
    """Nemenyi 事后（基于平均秩，配合 Kruskal-Wallis）。返回两两结果 + CLD。"""
    a = alpha or _CFG["alpha"]
    labels = list(groups.keys())
    flat = [x for g in groups.values() for x in g]
    N = len(flat)
    k = len(labels)
    if N < k or k < 2:
        return {"error": "Nemenyi 需 ≥2 组且每组有数据"}
    rk = _ranks(flat)
    grp_ranks = OrderedDict((l, []) for l in labels)
    flat_labels = []
    for l in labels:
        for _ in groups[l]:
            flat_labels.append(l)
    for lab, r_ in zip(flat_labels, rk):
        grp_ranks[lab].append(r_)
    Rbars = {l: mean(rs) for l, rs in grp_ranks.items()}
    n_per = {l: len(groups[l]) for l in labels}
    qcrit = _q_ppf(a, k, float("inf"))
    denom = math.sqrt(k * (k + 1) / (6.0 * N))
    out = []
    sig = {}
    for i in range(k):
        for j in range(i + 1, k):
            li, lj = labels[i], labels[j]
            diff = abs(Rbars[li] - Rbars[lj])
            se = denom * math.sqrt(1.0 / n_per[li] + 1.0 / n_per[lj]) if denom > 0 else None
            qstat = diff / se if (se and se > 0) else None
            sig[(li, lj)] = bool(qstat is not None and qstat > qcrit)
            out.append({"a": li, "b": lj, "mean_rank_diff": round(diff, 4),
                        "q_statistic": round(qstat, 4) if qstat is not None else None,
                        "q_critical": round(qcrit, 4),
                        "significant": bool(qstat is not None and qstat > qcrit)})
    means = {l: (mean(groups[l]) if groups[l] else None) for l in labels}
    cld = _compact_letters(labels, [means[l] for l in labels],
                           {(min(a_, b_), max(a_, b_)): s for (a_, b_), s in sig.items()})
    return {"method": "Nemenyi (studentized range on mean ranks)", "alpha": a,
            "comparisons": out, "compact_letters": cld}


def _scheffe(groups, mse, alpha):
    """Scheffe 事后（基于 F；与任意对比兼容）。需 MSE 与 k。返回两两结果。"""
    a = alpha or _CFG["alpha"]
    labels = list(groups.keys())
    k = len(labels)
    ns = {l: len(groups[l]) for l in labels}
    N = sum(ns.values())
    df2 = N - k
    fcrit = _f_ppf(1 - a, k - 1, df2)
    out = []
    means = {l: (mean(groups[l]) if groups[l] else None) for l in labels}
    for i in range(k):
        for j in range(i + 1, k):
            li, lj = labels[i], labels[j]
            ni, nj = ns[li], ns[lj]
            diff = (means[li] or 0) - (means[lj] or 0)
            se = math.sqrt(mse * (1.0 / ni + 1.0 / nj)) if mse and mse > 0 else None
            F = (diff ** 2) / (mse * (1.0 / ni + 1.0 / nj)) if (se and mse and mse > 0) else None
            fcrit_k = fcrit * (k - 1) if fcrit is not None else None
            out.append({"a": li, "b": lj, "mean_diff": round(diff, 4),
                        "F": round(F, 4) if F is not None else None,
                        "F_critical_scheffe": round(fcrit_k, 4) if fcrit_k is not None else None,
                        "significant": bool(F is not None and fcrit_k is not None and F > fcrit_k)})
    return {"method": "Scheffe", "alpha": a, "comparisons": out}


def _icc(matrix, model="2-1"):
    """组内相关 ICC。matrix: subjects(行) × raters(列) 数值矩阵。
    model: '1-1'(one-way random) / '2-1'(two-way random) / '3-1'(two-way mixed)。"""
    try:
        n = len(matrix)
        m = len(matrix[0]) if n else 0
        if n < 2 or m < 2:
            return {"error": "ICC 需 ≥2 个对象且 ≥2 个评分者"}
        means_row = [mean(r) for r in matrix]
        means_col = [mean(matrix[i][j] for i in range(n)) for j in range(m)]
        grand = mean(means_row)
        sst = sum((matrix[i][j] - grand) ** 2 for i in range(n) for j in range(m))
        ssa = m * sum((mr - grand) ** 2 for mr in means_row)
        ssb = n * sum((mc - grand) ** 2 for mc in means_col)
        sse = sst - ssa - ssb
        dfa, dfb, dfe = n - 1, m - 1, (n - 1) * (m - 1)
        msa = ssa / dfa if dfa else 0.0
        msb = ssb / dfb if dfb else 0.0
        mse = sse / dfe if dfe else 0.0
        if model == "1-1":
            icc = (msa - mse) / msa if msa else None
            f = msa / mse if mse else None
        elif model == "2-1":
            icc = (msa - mse) / (msa + (m - 1) * mse) if (msa + (m - 1) * mse) else None
            f = msa / mse if mse else None
        else:  # 3-1
            denom = msa + (m - 1) * mse + (m * (msb - mse) / n) if n else None
            icc = (msa - mse) / denom if denom else None
            f = msa / mse if mse else None
        p = _f_sf(f, dfa, dfe) if f is not None else None
        return {"model": model, "icc": _r4(icc), "F": round(f, 4) if f is not None else None,
                "df1": dfa, "df2": dfe, "p_value": _r4(p),
                "ms_subjects": round(msa, 4), "ms_raters": round(msb, 4),
                "ms_residual": round(mse, 4), "n_subjects": n, "n_raters": m,
                "interpretation": "ICC=%.3f（%s）" % (icc if icc is not None else 0,
                                                    _effect_label("eta2", icc) if icc is not None else "NA")}
    except Exception as e:
        return {"error": "ICC 计算失败: %s" % e}


def _cronbach_alpha(items):
    """Cronbach's α 信度。items: subjects(行) × 题项(列) 数值矩阵。"""
    try:
        n = len(items)
        k = len(items[0]) if n else 0
        if n < 2 or k < 2:
            return {"error": "Cronbach 需 ≥2 个样本且 ≥2 个题项"}
        item_vars = [stdev([items[i][j] for i in range(n)]) ** 2 for j in range(k)]
        total = [sum(items[i][j] for j in range(k)) for i in range(n)]
        total_var = stdev(total) ** 2
        if total_var == 0:
            return {"alpha": None, "note": "总分为常数，无法计算"}
        alpha = k / (k - 1.0) * (1 - sum(item_vars) / total_var)
        item_total = []
        for j in range(k):
            it = [items[i][j] for i in range(n)]
            m_it, m_total = mean(it), mean(total)
            s = sum((it[i] - m_it) * (total[i] - m_total) for i in range(n))
            denom = math.sqrt(sum((x - m_it) ** 2 for x in it) *
                              sum((x - m_total) ** 2 for x in total))
            item_total.append(round(s / denom, 4) if denom > 0 else None)
        return {"alpha": round(alpha, 4), "n_items": k, "n_subjects": n,
                "item_total_corr": item_total,
                "interpretation": "Cronbach α=%.3f（%s）" % (
                    alpha, "可接受(≥0.7)" if alpha >= 0.7 else ("需谨慎(0.6-0.7)" if alpha >= 0.6 else "偏低(<0.6)"))}
    except Exception as e:
        return {"error": "Cronbach 计算失败: %s" % e}


# ----- T1① 生存分析 -----
def _logrank(groups):
    """groups: {label: [(time, event_bool), ...]}。返回 log-rank 检验与各组 O/E。"""
    labels = list(groups.keys())
    all_times = sorted(set(t for lst in groups.values() for (t, e) in lst))
    O = {l: 0.0 for l in labels}
    E = {l: 0.0 for l in labels}
    for t in all_times:
        risk = {l: sum(1 for (tt, ee) in groups[l] if tt >= t) for l in labels}
        ev_t = sum(1 for l in labels for (tt, ee) in groups[l] if tt == t and ee == 1)
        n_risk = sum(risk.values())
        if n_risk == 0:
            continue
        for l in labels:
            O[l] += sum(1 for (tt, ee) in groups[l] if tt == t and ee == 1)
            E[l] += risk[l] * ev_t / n_risk
    chi = sum((O[l] - E[l]) ** 2 / E[l] for l in labels if E[l] > 0)
    df = max(len(labels) - 1, 1)
    p = _chi2_sf(chi, df)
    per_group = {l: {"observed": round(O[l], 2), "expected": round(E[l], 2),
                     "n": len(groups[l])} for l in labels}
    return {"method": "log-rank", "chi2": round(chi, 4), "df": df,
            "p_value": _r4(p), "significant": _sig(p), "per_group": per_group,
            "interpretation": "log-rank χ²=%.3f，p=%s，各组生存曲线%s（α=%g）" % (
                chi, _fmt_p(p), "差异显著" if _sig(p) else "未见显著差异", _CFG["alpha"])}


def _cox_ph(X, times, events, alpha=0.05):
    """Cox 比例风险回归（Breslow 近似，纯 Python）。X: n×p 设计矩阵(已标准化)。
    返回系数/HR/SE/95%CI/Wald p 与 concordance index。"""
    n = len(times)
    if n < 3 or not X or len(X[0]) == 0:
        return {"error": "Cox PH 需 ≥3 样本且至少 1 个特征"}
    p = len(X[0])
    means = [mean([X[i][j] for i in range(n)]) for j in range(p)]
    sds = [stdev([X[i][j] for i in range(n)]) or 1.0 for j in range(p)]
    Xs = [[(X[i][j] - means[j]) / sds[j] for j in range(p)] for i in range(n)]
    beta = [0.0] * p
    event_times = sorted(set(times[i] for i in range(n) if events[i] == 1))
    for _ in range(100):
        expeta = [math.exp(sum(beta[j] * Xs[i][j] for j in range(p))) for i in range(n)]
        U = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for tj in event_times:
            risk = [i for i in range(n) if times[i] >= tj]
            ej = [i for i in risk if events[i] == 1 and times[i] == tj]
            if not ej:
                continue
            s0 = sum(expeta[i] for i in risk)
            if s0 <= 0:
                continue
            s1 = [sum(Xs[i][j] * expeta[i] for i in risk) for j in range(p)]
            s2 = [[sum(Xs[i][j] * Xs[i][k] * expeta[i] for i in risk) for k in range(p)] for j in range(p)]
            d = len(ej)
            for idx_e in ej:
                for j in range(p):
                    U[j] += Xs[idx_e][j] - d * s1[j] / s0
                for j in range(p):
                    for k in range(p):
                        H[j][k] += d * (s2[j][k] / s0 - s1[j] * s1[k] / (s0 * s0))
        try:
            delta = gaussian_solve(H, U)
        except Exception:
            delta = [U[j] / (H[j][j] + 1e-8) for j in range(p)]
        beta = [beta[j] + delta[j] for j in range(p)]
        if max(abs(d) for d in delta) < 1e-6:
            break
    # 最终信息矩阵求逆得 SE
    expeta = [math.exp(sum(beta[j] * Xs[i][j] for j in range(p))) for i in range(n)]
    H = [[0.0] * p for _ in range(p)]
    for tj in event_times:
        risk = [i for i in range(n) if times[i] >= tj]
        ej = [i for i in risk if events[i] == 1 and times[i] == tj]
        if not ej:
            continue
        s0 = sum(expeta[i] for i in risk)
        if s0 <= 0:
            continue
        s1 = [sum(Xs[i][j] * expeta[i] for i in risk) for j in range(p)]
        s2 = [[sum(Xs[i][j] * Xs[i][k] * expeta[i] for i in risk) for k in range(p)] for j in range(p)]
        d = len(ej)
        for j in range(p):
            for k in range(p):
                H[j][k] += d * (s2[j][k] / s0 - s1[j] * s1[k] / (s0 * s0))
    invH = []
    for c in range(p):
        e = [1.0 if i == c else 0.0 for i in range(p)]
        try:
            col = gaussian_solve(H, e)
        except Exception:
            col = [0.0] * p
        invH.append(col)
    zc = _norm_ppf(0.5 + (1 - alpha) / 2.0)
    coefs = []
    risks = [sum(beta[j] * Xs[i][j] for j in range(p)) for i in range(n)]
    cidx = _concordance_index(times, risks, events)
    for j in range(p):
        se = math.sqrt(max(invH[j][j], 0.0))
        b = beta[j]
        hr = math.exp(b)
        lo = math.exp(b - zc * se) if zc else None
        hi = math.exp(b + zc * se) if zc else None
        z = b / se if se > 0 else 0.0
        pv = (_norm_sf(abs(z)) * 2.0) if se > 0 else None
        coefs.append({"coef": round(b, 4), "hr": round(hr, 4),
                      "se": round(se, 4), "ci95_hr": [round(lo, 4), round(hi, 4)] if lo is not None else None,
                      "wald_z": round(z, 4), "p_value": _r4(pv),
                      "significant": bool(pv is not None and pv < alpha)})
    return {"model": "Cox PH (Breslow)", "n": n, "n_events": int(sum(events)),
            "coefficients": coefs, "concordance_index": round(cidx, 4) if cidx is not None else None,
            "note": "HR>1 表示该特征升高风险；纯 Python Breslow 近似，样本大/强相关时建议 scipy lifelines 复核"}


def _concordance_index(times, risks, events):
    """C-index：对可比对的（事件 vs 更长寿/删失更晚）计算一致比例。"""
    n = len(times)
    concordant = 0.0
    tie = 0.0
    perm = 0
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] > times[i]:
                perm += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    tie += 1
    if perm == 0:
        return None
    return (concordant + 0.5 * tie) / perm


# ----- T1③ 自动化 EDA + Alerts -----
def _assoc_matrix(columns, rows):
    """两两关联矩阵：数值-数值→Pearson r；分类-分类→Cramér's V；混合→η²。"""
    num = _numeric_columns(columns, rows)
    cat = [c for c in columns if c not in num]
    cols = num + cat
    mat = {c: {} for c in cols}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            av = [to_float(r.get(a)) for r in rows]
            bv = [to_float(r.get(b)) for r in rows]
            if a in num and b in num:
                pairs = [(x, y) for x, y in zip(av, bv) if x is not None and y is not None]
                if len(pairs) >= 2:
                    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
                    r = pearson(xs, ys)
                    val = round(r, 4) if r == r else None
                    kind = "pearson_r"
                else:
                    val, kind = None, "pearson_r"
            elif a not in num and b not in num:
                ct, _, _ = _crosstab([str(r.get(a)) for r in rows], [str(r.get(b)) for r in rows])
                val = _cramers_v(ct)
                kind = "cramers_v"
            else:
                numc, catc = (a, b) if a in num else (b, a)
                eta = _eta_assoc([str(r.get(catc)) for r in rows], [to_float(r.get(numc)) for r in rows])
                val = eta
                kind = "eta_squared"
            if val is not None:
                mat[a][b] = {"value": val, "kind": kind}
                mat[b][a] = {"value": val, "kind": kind}
    return mat


def _crosstab(a, b):
    levels_a = sorted(set(a))
    levels_b = sorted(set(b))
    idx_a = {v: i for i, v in enumerate(levels_a)}
    idx_b = {v: i for i, v in enumerate(levels_b)}
    ct = [[0] * len(levels_b) for _ in range(len(levels_a))]
    for x, y in zip(a, b):
        if x in idx_a and y in idx_b:
            ct[idx_a[x]][idx_b[y]] += 1
    return ct, levels_a, levels_b


def cmd_profile(args):
    """自动化 EDA：类型推断 + Alerts 告警 + 关联矩阵 +（可选）train-test 漂移对比。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    alerts = []
    per_col = OrderedDict()
    for c in columns:
        vals = [r.get(c) for r in rows]
        n = len(vals)
        floats = [to_float(v) for v in vals]
        nonnull = [v for v in vals if v is not None and str(v).strip() != ""]
        miss = n - len(nonnull)
        miss_rate = miss / n if n else 0.0
        is_num = any(f is not None for f in floats)
        nun = [f for f in floats if f is not None]
        uniq = len(set(str(v) for v in nonnull))
        info = {"dtype": "numeric" if is_num else "categorical",
                "n": n, "missing": miss, "missing_rate": round(miss_rate, 4),
                "n_unique": uniq}
        if miss_rate > 0.2:
            alerts.append({"column": c, "severity": "high" if miss_rate > 0.5 else "medium",
                           "type": "high_missing", "message": "缺失率 %.1f%%" % (miss_rate * 100)})
        if is_num and len(nun) >= 2:
            m, s = mean(nun), (stdev(nun) if len(nun) > 1 else 0.0)
            info["mean"] = round(m, 4); info["std"] = round(s, 4)
            info["min"] = round(min(nun), 4); info["max"] = round(max(nun), 4)
            if s and s > 0:
                g = _normality_one(nun)
                info["skewness_ok"] = g.get("normal")
                sk = (sum((x - m) ** 3 for x in nun) / len(nun)) / (s ** 3) if s else 0.0
                info["skewness"] = round(sk, 4)
                if abs(sk) > 1.0:
                    alerts.append({"column": c, "severity": "medium", "type": "high_skew",
                                   "message": "偏度 %.2f（|偏度|>1）" % sk})
            zeros = sum(1 for x in nun if x == 0)
            if zeros / len(nun) > 0.5:
                alerts.append({"column": c, "severity": "low", "type": "many_zeros",
                               "message": "零值占比 %.1f%%" % (zeros / len(nun) * 100)})
        if uniq == 1:
            alerts.append({"column": c, "severity": "high", "type": "constant",
                           "message": "该列仅 1 个唯一值（常量，无信息量）"})
        elif not is_num and uniq > max(50, 0.5 * n):
            alerts.append({"column": c, "severity": "medium", "type": "high_cardinality",
                           "message": "高基数分类（%d 个唯一值，n=%d）" % (uniq, n)})
        per_col[c] = info
    assoc = _assoc_matrix(columns, rows)
    high_assoc = []
    for a in assoc:
        for b, v in assoc[a].items():
            if a >= b:
                continue
            if v["kind"] == "pearson_r" and abs(v["value"]) > 0.9:
                high_assoc.append({"a": a, "b": b, "value": v["value"], "kind": v["kind"]})
                alerts.append({"columns": [a, b], "severity": "medium", "type": "high_correlation",
                               "message": "%s 与 %s 相关系数 %.2f" % (a, b, v["value"])})
            elif v["kind"] in ("cramers_v", "eta_squared") and v["value"] > 0.8:
                high_assoc.append({"a": a, "b": b, "value": v["value"], "kind": v["kind"]})
                alerts.append({"columns": [a, b], "severity": "low", "type": "high_association",
                               "message": "%s~%s 关联强度 %.2f" % (a, b, v["value"])})
    rep = {"task": "profile", "rows": len(rows), "columns": columns,
           "per_column": per_col, "alerts": alerts, "n_alerts": len(alerts),
           "high_association_pairs": high_assoc, "association_matrix": assoc,
           "interpretation": "共 %d 条告警（高缺失/常量/高偏度/高基数/高相关），详见 alerts。" % len(alerts)}
    if args.compare:
        if not os.path.exists(args.compare):
            die("对比数据集不存在: %s" % args.compare)
        _, rows2 = load_rows(args.compare)
        drift = []
        for c in columns:
            if c not in [cc for cc in (rows2[0].keys() if rows2 else [])]:
                continue
            a1 = [to_float(r.get(c)) for r in rows if to_float(r.get(c)) is not None]
            a2 = [to_float(r.get(c)) for r in rows2 if to_float(r.get(c)) is not None]
            if len(a1) >= 2 and len(a2) >= 2:
                m1, m2 = mean(a1), mean(a2)
                s1 = stdev(a1)
                psi = _psi(a1, a2)
                drift.append({"column": c, "mean_base": round(m1, 4), "mean_compare": round(m2, 4),
                              "mean_shift": round(m2 - m1, 4), "psi": round(psi, 4)})
        rep["drift_compare"] = {"reference": args.input, "comparison": args.compare, "per_column": drift}
    if args.output:
        _atomic_write(args.output, json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        rep["output"] = args.output
    emit({"status": "ok", "task": "profile", "result": rep})


def _psi(a, b, bins=10):
    """人口稳定性指数（简化分位桶法）。"""
    try:
        qs = [min(a) + (max(a) - min(a)) * i / bins for i in range(bins + 1)]
        def dist(xs):
            return [sum(1 for x in xs if (qs[i] <= x < qs[i + 1]) or (i == bins - 1 and x >= qs[i])) / len(xs)
                    for i in range(bins)]
        da = dist(a); db = dist(b)
        s = 0.0
        for pa, pb in zip(da, db):
            pa = max(pa, 1e-4); pb = max(pb, 1e-4)
            s += (pb - pa) * math.log(pb / pa)
        return s
    except Exception:
        return None


# ----- T1② Table 1 -----
def cmd_table1(args):
    """论文 Table 1：分组基线描述（连续量 均值±SD / 中位数[IQR]；分类量 频数与%）+ 组间 p + SMD。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    group_col = args.group
    if group_col and group_col not in columns:
        die("分组列不存在: %s" % group_col)
    if group_col:
        groups = OrderedDict()
        for r in rows:
            gv = r.get(group_col)
            if gv is not None and str(gv).strip() != "":
                groups.setdefault(str(gv), []).append(r)
        if len(groups) < 2:
            die("分组需 ≥2 个非空水平")
    else:
        groups = {"ALL": rows}
    vars_spec = [v.strip() for v in args.vars.split(",") if v.strip()]
    if not vars_spec:
        die("需指定 --vars（逗号分隔的列名）")
    table = []
    overall_sig = {}
    for v in vars_spec:
        if v not in columns:
            die("变量不存在: %s" % v)
        coldata = {g: [to_float(r.get(v)) for r in rs] for g, rs in groups.items()}
        is_num = any(x is not None for x in coldata[list(coldata.keys())[0]])
        entry = {"variable": v, "type": "continuous" if is_num else "categorical"}
        if is_num:
            grp_stats = {}
            for g, xs in coldata.items():
                xs = [x for x in xs if x is not None]
                if xs:
                    m = mean(xs); sd = stdev(xs) if len(xs) > 1 else 0.0
                    med = median(xs); q1 = _quantile(xs, 0.25); q3 = _quantile(xs, 0.75)
                    grp_stats[g] = {"n": len(xs), "mean": round(m, 4), "sd": round(sd, 4),
                                    "median": round(med, 4), "q1": round(q1, 4), "q3": round(q3, 4)}
                else:
                    grp_stats[g] = {"n": 0}
            entry["groups"] = grp_stats
            # 组间检验：≥3 组 ANOVA，2 组 t
            vals_by_g = {g: [x for x in coldata[g] if x is not None] for g in groups}
            if len(groups) == 2:
                gl = list(groups.keys())
                _, p = _t_test_two(vals_by_g[gl[0]], vals_by_g[gl[1]])
                entry["p_value"] = _r4(p)
            elif len(groups) > 2:
                _, p = _anova_F([vals_by_g[g] for g in groups])
                entry["p_value"] = _r4(p)
            # SMD：对照组(第一组) vs 其余，或 ALL 单组不适用
            if len(groups) >= 2:
                gl = list(groups.keys())
                smds = {}
                for g in gl[1:]:
                    smds[g] = _smd([x for x in coldata[gl[0]] if x is not None],
                                    [x for x in coldata[g] if x is not None])
                entry["smd_vs_ref"] = smds
        else:
            levels = sorted(set(str(r.get(v)) for rs in groups.values() for r in rs
                                if r.get(v) is not None and str(r.get(v)).strip() != ""))
            grp_stats = {}
            for g, rs in groups.items():
                n = len([r for r in rs if r.get(v) is not None and str(r.get(v)).strip() != ""])
                cnt = Counter(str(r.get(v)) for r in rs if r.get(v) is not None and str(r.get(v)).strip() != "")
                grp_stats[g] = {"n": n, "levels": {lv: {"count": cnt.get(lv, 0),
                                                         "pct": round(100.0 * cnt.get(lv, 0) / n, 2) if n else 0.0}
                                                    for lv in levels}}
            entry["groups"] = grp_stats
            # 卡方
            ct, _, _ = _crosstab([str(r.get(v)) for rs in groups.values() for r in rs
                                  if r.get(v) is not None and str(r.get(v)).strip() != ""],
                                 [str(r.get(group_col)) for rs in groups.values() for r in rs
                                  if r.get(v) is not None and str(r.get(v)).strip() != ""])
            _, p = _chi2_from_ct(ct)
            entry["p_value"] = _r4(p)
        table.append(entry)
        overall_sig[v] = entry.get("p_value")
    rep = {"task": "table1", "group": group_col, "groups": list(groups.keys()),
           "n_per_group": {g: len(rs) for g, rs in groups.items()},
           "table": table,
           "note": "连续量给出 均值±SD（偏态时可参考 median[IQR]）；p 值为组间检验；SMD>0.1 提示组间不均衡"}
    if args.output:
        _write_table1(rep, args.output)
        rep["output"] = args.output
    emit({"status": "ok", "task": "table1", "result": rep})


def _quantile(xs, q):
    """分位数（q 取 0-1，线性插值）；空输入返回 None。复用 quantile。"""
    if not xs:
        return None
    return quantile(sorted(xs), q)


def _t_test_two(a, b):
    try:
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return None, None
        m1, m2 = mean(a), mean(b)
        v1, v2 = stdev(a) ** 2, stdev(b) ** 2
        sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        se = sp * math.sqrt(1.0 / n1 + 1.0 / n2)
        t = (m1 - m2) / se if se > 0 else 0.0
        df = n1 + n2 - 2
        return t, _t_two_sided_p(abs(t), df)
    except Exception:
        return None, None


def _anova_F(groups):
    try:
        allv = [x for g in groups for x in g]
        grand = mean(allv)
        sst = sum((x - grand) ** 2 for x in allv)
        ssa = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
        sse = sst - ssa
        k = len(groups); n = len(allv)
        dfa, dfe = k - 1, n - k
        F = (ssa / dfa) / (sse / dfe) if dfe > 0 and sse > 0 else float("inf")
        return F, _f_sf(F, dfa, dfe)
    except Exception:
        return None, None


def _chi2_from_ct(ct):
    try:
        n = sum(sum(r) for r in ct)
        if n == 0:
            return None, None
        row_t = [sum(r) for r in ct]
        col_t = [sum(ct[i][j] for i in range(len(ct))) for j in range(len(ct[0]))]
        chi = 0.0
        for i in range(len(ct)):
            for j in range(len(ct[0])):
                exp = row_t[i] * col_t[j] / n
                if exp > 0:
                    chi += (ct[i][j] - exp) ** 2 / exp
        df = (len(ct) - 1) * (len(ct[0]) - 1)
        return chi, _chi2_sf(chi, df)
    except Exception:
        return None, None


def _write_table1(rep, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        lines = ["variable,type,group,n,stat1,stat2,p_value,smd\n"]
        for e in rep["table"]:
            for g, gs in e.get("groups", {}).items():
                if e["type"] == "continuous":
                    lines.append("%s,continuous,%s,%s,%.4f,%.4f,%s,%s\n" % (
                        e["variable"], g, gs.get("n", ""), gs.get("mean", 0), gs.get("sd", 0),
                        e.get("p_value"), e.get("smd_vs_ref", {}).get(g, "")))
                else:
                    lines.append("%s,categorical,%s,%s,%s,%s,%s,\n" % (
                        e["variable"], g, gs.get("n", ""),
                        ";".join("%s:%d(%.1f%%)" % (lv, d["count"], d["pct"]) for lv, d in gs.get("levels", {}).items()),
                        "", e.get("p_value")))
        _atomic_write(path, "".join(lines))
    elif ext in (".md", ".markdown"):
        lines = ["# Table 1 基线特征\n", "| 变量 | 类型 | " +
                 " | ".join(rep["groups"]) + " | p |\n",
                 "| --- | --- | " + " | ".join(["---"] * len(rep["groups"])) + " | --- |\n"]
        for e in rep["table"]:
            cells = []
            for g in rep["groups"]:
                gs = e.get("groups", {}).get(g, {})
                if e["type"] == "continuous":
                    cells.append("%s±%s" % (gs.get("mean", "-"), gs.get("sd", "-")))
                else:
                    cells.append("; ".join("%s:%d(%.1f%%)" % (lv, d["count"], d["pct"])
                                           for lv, d in gs.get("levels", {}).items()))
            lines.append("| %s | %s | %s | %s |\n" % (e["variable"], e["type"], " | ".join(cells), e.get("p_value")))
        _atomic_write(path, "".join(lines))
    else:
        _atomic_write(path, json.dumps(rep, ensure_ascii=False, indent=2, default=str))


# ----- T2⑦ 数据质量契约 validate -----
def cmd_validate(args):
    """数据质量契约：按规则套件校验，输出 Data Docs 式报告。规则可来自 --rules(JSON) 或内联 --rule。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    rules = []
    if args.rules:
        if not os.path.exists(args.rules):
            die("规则文件不存在: %s" % args.rules)
        with open(args.rules, "r", encoding="utf-8") as f:
            rules = json.load(f)
    elif args.rule:
        for r in args.rule:
            # 形如 column:min_max:0:120 / column:not_null_rate:0.95 /
            #       column:unique / column:allowed:M,F / column:dtype:numeric
            parts = r.split(":")
            if len(parts) < 2:
                continue
            col, chk = parts[0], parts[1]
            item = {"column": col, "check": chk}
            if chk == "allowed":
                item["values"] = [x for x in parts[2].split(",") if x != ""] if len(parts) > 2 else []
            elif chk == "dtype":
                item["type"] = parts[2] if len(parts) > 2 else "numeric"
            elif chk == "min_max":
                item["min"] = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
                item["max"] = float(parts[3]) if len(parts) > 3 and parts[3] != "" else None
            else:  # not_null_rate / unique 等数值型阈值
                item["min"] = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
            rules.append(item)
    if not rules:
        die("需提供 --rules(规则文件) 或 --rule(内联规则)")
    results = []
    for rule in rules:
        col = rule.get("column")
        if col not in columns:
            results.append({"column": col, "check": rule.get("check"), "passed": False,
                            "message": "列不存在"})
            continue
        vals = [r.get(col) for r in rows]
        n = len(vals)
        nonnull = [v for v in vals if v is not None and str(v).strip() != ""]
        floats = [to_float(v) for v in nonnull]
        numeric = [f for f in floats if f is not None]
        passed = True
        msg = "通过"
        chk = rule.get("check")
        if chk == "not_null_rate":
            rate = len(nonnull) / n if n else 0.0
            thr = rule.get("min", 0.95)
            passed = rate >= thr
            msg = "非空率 %.3f（阈值 %.3f）" % (rate, thr)
        elif chk == "min_max":
            mn, mx = rule.get("min"), rule.get("max")
            bad = [x for x in numeric if (mn is not None and x < mn) or (mx is not None and x > mx)]
            passed = len(bad) == 0
            msg = "越界 %d 条（范围 %s~%s）" % (len(bad), mn, mx)
        elif chk == "unique":
            passed = len(nonnull) == len(set(str(v) for v in nonnull))
            msg = "唯一值 %d / %d" % (len(set(str(v) for v in nonnull)), len(nonnull))
        elif chk == "allowed":
            allowed = set(str(x) for x in rule.get("values", []))
            bad = [str(v) for v in nonnull if str(v) not in allowed]
            passed = len(bad) == 0
            msg = "非法取值 %d 条" % len(bad)
        elif chk == "dtype":
            is_num = len(numeric) > 0
            passed = (rule.get("type") == "numeric") == is_num
            msg = "推断类型 %s" % ("numeric" if is_num else "categorical")
        else:
            msg = "未知检查类型: %s" % chk
            passed = False
        results.append({"column": col, "check": chk, "passed": passed, "message": msg})
    n_pass = sum(1 for r in results if r["passed"])
    emit({"status": "ok", "task": "validate", "result": {
        "input": args.input, "n_rules": len(results), "n_passed": n_pass,
        "n_failed": len(results) - n_pass, "all_passed": n_pass == len(results),
        "checks": results}})


# ----- T1① 生存分析命令 -----
def cmd_survival(args):
    """生存分析：log-rank 组间检验 或 Cox PH 风险回归（支持删失标记）。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    time_col = args.time
    event_col = args.event
    if time_col not in columns or event_col not in columns:
        die("时间列/事件列不存在: %s / %s" % (time_col, event_col))
    times = [to_float(r.get(time_col)) for r in rows]
    ev_raw = [r.get(event_col) for r in rows]
    events = [1 if (str(e).strip() in ("1", "True", "true", "TRUE", "是", "yes", "Y")) else 0 for e in ev_raw]
    pairs = [(t, e) for t, e in zip(times, events) if t is not None]
    if args.mode == "logrank":
        group_col = args.group
        if not group_col or group_col not in columns:
            die("log-rank 需指定 --group（分组列）")
        groups = OrderedDict()
        for (t, e), r in zip(pairs, rows):
            gv = r.get(group_col)
            if gv is not None and str(gv).strip() != "":
                groups.setdefault(str(gv), []).append((t, e == 1))
        if len(groups) < 2:
            die("log-rank 需 ≥2 组")
        res = _logrank(groups)
        emit({"status": "ok", "task": "survival", "mode": "logrank", "result": res})
        return
    feats = [c.strip() for c in args.features.split(",") if c.strip()] if args.features else []
    if not feats:
        die("Cox PH 需 --features（逗号分隔的数值特征列）")
    for f in feats:
        if f not in columns:
            die("特征列不存在: %s" % f)
    X = []
    tt = []
    ev = []
    for r, (t, e) in zip(rows, pairs):
        if t is None:
            continue
        X.append([to_float(r.get(f)) for f in feats])
        tt.append(t)
        ev.append(e)
    # 列均值填补缺失（仅训练集精神可，但 Cox 此处整体填补以保证矩阵完整；缺失比例高时提示）
    if X:
        ncol = len(X[0])
        col_means = []
        for j in range(ncol):
            vals = [row[j] for row in X if row[j] is not None]
            col_means.append(mean(vals) if vals else 0.0)
        X = [[(v if v is not None else col_means[j]) for j, v in enumerate(row)] for row in X]
    if len(set(ev)) < 2:
        die("Cox PH 需同时含事件(1)与删失(0)")
    res = _cox_ph(X, tt, ev, args.alpha or _CFG["alpha"])
    if args.output:
        _atomic_write(args.output, json.dumps(res, ensure_ascii=False, indent=2, default=str))
        res["output"] = args.output
    emit({"status": "ok", "task": "survival", "mode": "cox", "features": feats, "result": res})


# ----- T2⑥ 预测建模：工程化（compare / 重要性 / SMOTE / 保存管线） -----
def _pr_auc(yt, scores):
    """二分类 PR-AUC（按预测概率降序梯形积分）。yt: 0/1 列表。"""
    n = len(yt); order = sorted(range(n), key=lambda i: -scores[i])
    tp = sum(yt)
    if tp == 0 or (n - tp) == 0:
        return None
    tp_c = fp_c = 0; prec_prev = 1.0; rec_prev = 0.0; area = 0.0
    for i in order:
        if yt[i] == 1:
            tp_c += 1
        else:
            fp_c += 1
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 1.0
        rec = tp_c / tp
        area += (rec - rec_prev) * (prec + prec_prev) / 2.0
        prec_prev, rec_prev = prec, rec
    return area


def _classification_metrics(yt, pred, yproba, classes):
    """输出混淆矩阵、各类 P/R/F1、macro/weighted 聚合、二分类 ROC-AUC 与 PR-AUC。"""
    k = len(classes)
    cm = [[0] * k for _ in range(k)]
    for i in range(len(yt)):
        cm[yt[i]][pred[i]] += 1
    per = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k) if r != c)
        fn = sum(cm[c][r] for r in range(k) if r != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sup = sum(cm[r][c] for r in range(k))
        per.append({"class": classes[c], "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": sup})
    macro_p = round(mean([x["precision"] for x in per]), 4)
    macro_r = round(mean([x["recall"] for x in per]), 4)
    macro_f = round(mean([x["f1"] for x in per]), 4)
    tot = sum(x["support"] for x in per)
    w_p = round(sum(x["precision"] * x["support"] for x in per) / tot, 4) if tot else 0.0
    w_r = round(sum(x["recall"] * x["support"] for x in per) / tot, 4) if tot else 0.0
    w_f = round(sum(x["f1"] * x["support"] for x in per) / tot, 4) if tot else 0.0
    acc = round(sum(cm[c][c] for c in range(k)) / tot, 4) if tot else 0.0
    m = {"accuracy": acc, "precision_macro": macro_p, "recall_macro": macro_r, "f1_macro": macro_f,
         "precision_weighted": w_p, "recall_weighted": w_r, "f1_weighted": w_f,
         "per_class": per, "confusion_matrix": cm}
    if k == 2 and yproba is not None:
        try:
            pos = [yproba[i] for i in range(len(yt)) if yt[i] == 1]
            neg = [yproba[i] for i in range(len(yt)) if yt[i] == 0]
            auc = (sum(1.0 for p in pos for nn in neg if p > nn) +
                   0.5 * sum(1.0 for p in pos for nn in neg if p == nn))
            auc /= (len(pos) * len(neg)) if (pos and neg) else 1.0
            m["roc_auc"] = round(auc, 4)
            pra = _pr_auc(yt, yproba)
            if pra is not None:
                m["pr_auc"] = round(pra, 4)
        except Exception:
            pass
    return m


def _regression_metrics(yt, pred):
    n = len(yt)
    sq = sum((pred[i] - yt[i]) ** 2 for i in range(n))
    ae = sum(abs(pred[i] - yt[i]) for i in range(n))
    my = mean(yt); sst = sum((v - my) ** 2 for v in yt)
    r2 = 1 - sq / sst if sst > 0 else None
    return {"mse": round(sq / n, 4), "rmse": round(math.sqrt(sq / n), 4),
            "mae": round(ae / n, 4), "r2": round(r2, 4) if r2 is not None else None}


def _stratified_kfold(y, k, seed=0):
    """分层抽样 k 折，返回 [(train_idx, test_idx), ...]。"""
    rng = random.Random(seed)
    bycls = {}
    for i, yl in enumerate(y):
        bycls.setdefault(yl, []).append(i)
    for c in bycls:
        rng.shuffle(bycls[c])
    folds = [[] for _ in range(k)]
    for c, idxs in bycls.items():
        for j, ii in enumerate(idxs):
            folds[j % k].append(ii)
    return [([ii for ff in range(k) if ff != f for ii in folds[ff]], folds[f]) for f in range(k)]


def _build_enc_vec(data, feats, train_idx):
    """基于给定训练索引构建 编码(enc) 与 向量化函数(vec)，供单次切分与 CV 复用（防泄露）。"""
    num_feats = [f for f in feats if all(data[i][1][f][1] is not None for i in train_idx)]
    enc = {}
    for f in num_feats:
        vals = [data[i][1][f][1] for i in train_idx if data[i][1][f][1] is not None]
        m = mean(vals) if vals else 0.0
        s = stdev(vals) if len(vals) > 1 else 0.0
        enc[f] = (m, s if s and s > 0 else 1.0)
    for f in (f for f in feats if f not in num_feats):
        cats = sorted(set(str(data[i][1][f][0]) for i in train_idx if data[i][1][f][0] is not None))
        enc[f] = {c: j for j, c in enumerate(cats)}

    def vec(i):
        v = [1.0]
        for f in num_feats:
            fv = data[i][1][f][1]
            m, s = enc[f]
            v.append((fv - m) / s if fv is not None else 0.0)
        for f in (ff for ff in feats if ff not in num_feats):
            raw = data[i][1][f][0]
            v.extend(1.0 if (raw is not None and str(raw) in enc[f]) else 0.0 for _ in range(len(enc[f])))
        return v

    return enc, vec, num_feats


def _aggregate_cv(folds_metrics):
    """聚合多折指标：对每个标量数值键给出 mean±std（跳过 per_class/confusion_matrix 等非标量）。"""
    keys = set()
    for m in folds_metrics:
        for kk, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(kk)
    agg = {}
    for kk in sorted(keys):
        vals = [m[kk] for m in folds_metrics if kk in m and m[kk] is not None]
        if vals:
            agg[kk] = {"mean": round(mean(vals), 4),
                       "std": round(stdev(vals) if len(vals) > 1 else 0.0, 4)}
    return agg


def _predict_core(data, feats, train, test, enc, vec, y, task, model, args, smote=False,
                  importance=False, save_path=None):
    """predict 核心：拟合 + 指标；可选 SMOTE 过采样、置换重要性、保存 pipeline。"""
    num_feats = [f for f in feats if all(data[i][1][f][1] is not None for i in train)]
    cat_feats = [f for f in feats if f not in num_feats]
    knn_k = args.k if getattr(args, "k", None) is not None else 5
    if model == "knn":
        if knn_k < 1:
            die("knn 近邻数 --k 必须 ≥1")
        if knn_k > len(train):
            die("knn 近邻数 --k(%d) 超过训练样本数(%d)" % (knn_k, len(train)))
    if smote and task == "classification":
        # 少数类过采样（训练集内，防泄露）
        bycls = {}
        for i in train:
            bycls.setdefault(y[i], []).append(i)
        if len(bycls) == 2:
            maj, mino = max(bycls, key=lambda c: len(bycls[c])), min(bycls, key=lambda c: len(bycls[c]))
            need = len(bycls[maj]) - len(bycls[mino])
            rng = random.Random(args.seed or 0)
            add = [rng.choice(bycls[mino]) for _ in range(need)]
            # 复制样本（保留原始向量，避免重复增强引入噪声）
            train = train + add
    Xtr = [vec(i) for i in train]
    Xte = [vec(i) for i in test]
    d = len(Xtr[0])
    coef = None
    pred = [None] * len(test)
    yproba = [None] * len(test) if task == "classification" else None
    backend = None
    if HAS_SKLEARN and model in ("linear", "logistic", "knn"):
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            Xtrn = np.array(Xtr, dtype=float); Xten = np.array(Xte, dtype=float)
            if task == "regression" and model == "linear":
                mdl = LinearRegression().fit(Xtrn, [y[i] for i in train])
                pred = mdl.predict(Xten).tolist(); backend = "sklearn"
            elif task == "classification" and model == "logistic":
                mdl = LogisticRegression(max_iter=2000).fit(Xtrn, [y[i] for i in train])
                proba = mdl.predict_proba(Xten)
                pred = [int(mdl.classes_[int(np.argmax(p))]) for p in proba]
                coef = {feats[j]: round(float(mdl.coef_[0][j + 1]), 4) for j in range(len(feats))}
                yproba = [float(p[1]) for p in proba] if len(proba[0]) == 2 else [None] * len(proba)
                backend = "sklearn"
            elif model == "knn":
                if task == "regression":
                    mdl = KNeighborsRegressor(n_neighbors=knn_k).fit(Xtrn, [y[i] for i in train])
                    pred = mdl.predict(Xten).tolist()
                else:
                    mdl = KNeighborsClassifier(n_neighbors=knn_k).fit(Xtrn, [y[i] for i in train])
                    pred = mdl.predict(Xten).tolist()
                    proba = mdl.predict_proba(Xten)
                    yproba = [float(p[1]) if len(p) > 1 else float(p[0]) for p in proba]
                backend = "sklearn"
            else:
                backend = None
        except Exception:
            backend = None
    if backend is None:
        backend = "pure_python"
        if model == "knn":
            k = knn_k
            for ti, xi in enumerate(Xte):
                dist = sorted((sum((a - b) ** 2 for a, b in zip(xi, Xtr[j])), j) for j in range(len(Xtr)))
                nn = dist[:k]
                if task == "regression":
                    pred[ti] = mean([y[train[j]] for _, j in nn])
                else:
                    cnt = {}
                    for _, j in nn:
                        cnt[y[train[j]]] = cnt.get(y[train[j]], 0) + 1
                    pred[ti] = max(cnt, key=cnt.get)
                    if k and len(cnt) == 2:
                        yproba[ti] = cnt.get(1, 0) / k
        elif task == "regression" and model == "linear":
            XtX = [[sum(Xtr[i][a] * Xtr[i][b] for i in range(len(Xtr))) for b in range(d)] for a in range(d)]
            Xty = [sum(Xtr[i][a] * y[train[i]] for i in range(len(Xtr))) for a in range(d)]
            beta = gaussian_solve(XtX, Xty)
            if beta is None:
                for a in range(d):
                    XtX[a][a] += 1e-6
                beta = gaussian_solve(XtX, Xty)
            coef = {("(intercept)" if j == 0 else feats[j - 1]): round(beta[j], 4) for j in range(d)}
            pred = [sum(beta[a] * xi[a] for a in range(d)) for xi in Xte]
        elif task == "classification" and model == "logistic":
            K = len(set(y))
            W = [[0.0] * d for _ in range(K)]
            lr = 0.1
            yb = [[1.0 if y[train[i]] == c else 0.0 for i in range(len(train))] for c in range(K)]
            for c in range(K):
                w = [0.0] * d
                for _ in range(800):
                    grad = [0.0] * d
                    for i, ii in enumerate(train):
                        z = sum(w[a] * Xtr[i][a] for a in range(d))
                        err = _sigmoid(z) - yb[c][i]
                        for a in range(d):
                            grad[a] += err * Xtr[i][a]
                    for a in range(d):
                        w[a] -= lr * grad[a] / len(train)
                W[c] = w
            coef = {feats[j - 1]: round(float(W[0][j]), 4) for j in range(1, d)}
            for ti, xi in enumerate(Xte):
                ps = [_sigmoid(sum(W[c][a] * xi[a] for a in range(d))) for c in range(K)]
                pred[ti] = max(range(K), key=lambda c: ps[c])
                yproba[ti] = ps[1] if K == 2 else None
        else:
            return {"error": "模型不支持：task=%s model=%s" % (task, model)}

    yt = [y[i] for i in test]
    out = {"task": task, "model": model, "backend": backend, "smote": smote,
           "n_train": len(train), "n_test": len(test),
           "n_features": len(feats), "numeric_features": num_feats, "categorical_features": cat_feats,
           "coefficients": coef,
           "leakage_guard": "标准化/缺失/独热编码仅基于训练集拟合后应用到验证集"}

    def _metric(ps, yt):
        if task == "regression":
            sq = sum((p - yt[i]) ** 2 for i, p in enumerate(ps))
            m_yt = mean(yt)
            ss_tot = sum((v - m_yt) ** 2 for v in yt)
            return 1 - sq / ss_tot if ss_tot > 0 else 0.0
        correct = sum(1 for i in range(len(yt)) if ps[i] == yt[i])
        return correct / len(yt)

    classes = sorted(set(yt))
    if task == "regression":
        out["metrics"] = _regression_metrics(yt, pred)
    else:
        out["metrics"] = _classification_metrics(yt, pred, yproba, classes)
    if importance:
        rng = random.Random((args.seed or 0) + 1)
        n_rep = 5
        base = _metric(pred, yt)
        imp = {}
        for j in range(len(feats)):
            drops = []
            for _ in range(n_rep):
                Xs = [row[:] for row in Xte]
                col = [r[j + 1] for r in Xs]
                rng.shuffle(col)
                for i in range(len(Xs)):
                    Xs[i][j + 1] = col[i]
                ps = [sum(coef[a] * Xs[i][a] for a in range(d)) if (backend == "pure_python" and coef and task == "regression")
                      else _pp_pred(backend, coef, Xs[i], y, task, args, feats, num_feats, cat_feats, enc)
                      for i in range(len(Xs))]
                drops.append(_metric(ps, yt))
            imp[feats[j]] = round(base - mean(drops), 4)
        out["permutation_importance"] = imp
    if save_path:
        pipe = {"task": task, "model": model, "features": feats,
                "numeric_features": num_feats, "categorical_features": cat_feats,
                "encoding": {f: (list(enc[f]) if f in num_feats else enc[f]) for f in feats},
                "backend": backend, "seed": args.seed or 0}
        _atomic_write(save_path, json.dumps(pipe, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        out["saved_pipeline"] = save_path
    return out


def _pp_pred(backend, coef, xi, y, task, args, feats, num_feats, cat_feats, enc):
    """predict 阶段用的单样本预测（重要性重算时调用）。"""
    if task == "regression" and coef:
        return sum(coef.get(("(intercept)" if a == 0 else feats[a - 1]), 0.0) * xi[a] for a in range(len(xi)))
    if task == "classification" and coef:
        z = sum(coef.get(feats[a - 1], 0.0) * xi[a] for a in range(1, len(xi)))
        return 1 if _sigmoid(z) > 0.5 else 0
    return xi[1]


def cmd_predict(args):
    """预测建模入口：支持 --compare 多模型排名 / --importance 置换重要性 / --smote / --save-pipeline。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    target = args.target
    if target not in columns:
        die("目标变量不存在: %s" % target)
    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    if not feats:
        die("需指定 --features（逗号分隔的特征列）")
    for c in feats:
        if c not in columns:
            die("特征列不存在: %s" % c)
    data = []
    for r in rows:
        yraw = r.get(target)
        if yraw is None or str(yraw).strip() == "":
            continue
        fdict = {}
        for f in feats:
            raw = r.get(f)
            fdict[f] = (raw, to_float(raw))
        data.append((yraw, fdict))
    if len(data) < 10:
        die("有效样本不足（需 ≥10）")
    yvals = [d[0] for d in data]
    all_num = all(to_float(v) is not None for v in yvals)
    if args.task:
        task = args.task
    else:
        if all_num:
            uniq = set(str(v) for v in yvals)
            integral = all(float(v).is_integer() for v in yvals)
            task = "classification" if (len(uniq) <= 10 and integral) else "regression"
        else:
            task = "classification"
    if task == "regression":
        y = [to_float(v) for v in yvals]
    else:
        classes = sorted(set(str(v) for v in yvals))
        c2i = {c: i for i, c in enumerate(classes)}
        y = [c2i[str(v)] for v in yvals]
        if len(classes) < 2:
            die("分类任务需 ≥2 个类别")
    model = args.model or "auto"
    if model == "auto":
        model = "linear" if task == "regression" else "logistic"
    test_ratio = args.split if args.split is not None else 0.25
    if not (0.0 < test_ratio < 1.0):
        die("--split 切分比例必须介于 0 与 1 之间（不含端点）")
    rng = random.Random(args.seed or 0)
    idxs = list(range(len(data)))
    rng.shuffle(idxs)
    ntr = max(1, int(len(idxs) * (1 - test_ratio)))
    if task == "classification":
        bycls = {}
        for i in idxs:
            bycls.setdefault(y[i], []).append(i)
        train, test = [], []
        for c, cidx in bycls.items():
            rng.shuffle(cidx)
            k = max(1, int(len(cidx) * (1 - test_ratio)))
            train += cidx[:k]; test += cidx[k:]
    else:
        train, test = idxs[:ntr], idxs[ntr:]
    cv = args.cv or 0
    if cv and cv >= 2:
        if cv > len(data):
            die("--cv 折数(%d) 不能超过样本量(%d)" % (cv, len(data)))
        folds = _stratified_kfold(y, cv, args.seed or 0)
        if args.compare:
            models = ["linear", "knn"] if task == "regression" else ["logistic", "knn"]
            cv_rank = {}
            for mdl in models:
                per = []
                for tr, te in folds:
                    enc_f, vec_f, _ = _build_enc_vec(data, feats, tr)
                    o = _predict_core(data, feats, tr, te, enc_f, vec_f, y, task, mdl, args,
                                     smote=bool(args.smote))
                    per.append(o.get("metrics", {}))
                cv_rank[mdl] = _aggregate_cv(per)
            emit({"status": "ok", "task": "predict", "mode": "compare_cv", "task_type": task,
                  "n_folds": cv, "ranking_cv": cv_rank,
                  "best_by_accuracy" if task == "classification" else "best_by_r2":
                      max(cv_rank, key=lambda m: cv_rank[m].get("accuracy" if task == "classification" else "r2", {}).get("mean", 0))})
            return
        per = []
        for tr, te in folds:
            enc_f, vec_f, _ = _build_enc_vec(data, feats, tr)
            o = _predict_core(data, feats, tr, te, enc_f, vec_f, y, task, model, args,
                             smote=bool(args.smote), importance=bool(args.importance))
            per.append(o.get("metrics", {}))
        emit({"status": "ok", "task": "predict", "mode": "cv", "task_type": task,
              "model": model, "n_folds": cv, "cv_metrics": _aggregate_cv(per),
              "note": "k 折交叉验证（分层）；每折训练集内独立标准化/SMOTE，无泄露"})
        return
    # 单次切分
    enc, vec, _ = _build_enc_vec(data, feats, train)
    if args.compare:
        models = ["linear", "knn"] if task == "regression" else ["logistic", "knn"]
        ranking = {}
        for mdl in models:
            o = _predict_core(data, feats, train, test, enc, vec, y, task, mdl, args)
            ranking[mdl] = o.get("metrics", {})
        emit({"status": "ok", "task": "predict", "mode": "compare", "task_type": task,
              "n_train": len(train), "n_test": len(test),
              "ranking": ranking,
              "best_by_accuracy" if task == "classification" else "best_by_r2":
                  max(ranking, key=lambda m: ranking[m].get("accuracy" if task == "classification" else "r2", 0))})
        return
    out = _predict_core(data, feats, train, test, enc, vec, y, task, model, args,
                        smote=bool(args.smote), importance=bool(args.importance),
                        save_path=args.save_pipeline)
    emit({"status": "ok", "task": "predict", "result": out})


# ----- stats 扩展：Dunnett/Nemenyi/Scheffe/CLD/ICC/Cronbach/ANCOVA/Mixed/Mediation -----
def _ancova(groups_factor, cov, value, rows):
    """ANCOVA：单因素 + 协变量校正。groups_factor: 分组列；cov: 协变量列；value: 因变量。"""
    try:
        data = []
        for r in rows:
            g = r.get(groups_factor); y = to_float(r.get(value)); x = to_float(r.get(cov))
            if g is not None and y is not None and x is not None:
                data.append((str(g), x, y))
        if len(data) < 4:
            return {"error": "ANCOVA 样本不足"}
        # 总回归
        xs = [d[1] for d in data]; ys = [d[2] for d in data]
        gmeans = {g: mean([d[2] for d in data if d[0] == g]) for g in set(d[0] for d in data)}
        # 协变量斜率（合并，组内公共斜率）
        mx = mean(xs); my = mean(ys)
        b = sum((d[1] - mx) * (d[2] - my) for d in data) / sum((d[1] - mx) ** 2 for d in data)
        # 残差 = y - b*x
        resid = [d[2] - b * d[1] for d in data]
        # 在残差上做单因素 ANOVA
        pairs = [(d[0], r) for d, r in zip(data, resid)]
        oneway = _one_way_anova(pairs)
        adj_means = {}
        for g in set(d[0] for d in data):
            gx = mean([d[1] for d in data if d[0] == g])
            adj_means[g] = round(gmeans[g] - b * (gx - mx), 4)
        return {"model": "ANCOVA (one-factor + covariate)", "covariate_slope": round(b, 4),
                "adjusted_means": adj_means, "anova_residualized": oneway}
    except Exception as e:
        return {"error": "ANCOVA 失败: %s" % e}


def _mixed_anova(within, between, subject, value, rows):
    """混合 ANOVA：被试内因子 + 被试间因子 + 被试 ID。简化：两组内×组间双因素，含被试随机效应。"""
    try:
        # 组织为 long：subject, between_group, within_level, value
        recs = []
        for r in rows:
            s = r.get(subject); b = r.get(between); w = r.get(within); v = to_float(r.get(value))
            if s is not None and b is not None and w is not None and v is not None:
                recs.append((str(s), str(b), str(w), v))
        if len(recs) < 4:
            return {"error": "混合 ANOVA 样本不足"}
        bs = sorted(set(x[1] for x in recs))
        ws = sorted(set(x[2] for x in recs))
        subs = sorted(set(x[0] for x in recs))
        # 组间效应
        btwn_groups = OrderedDict()
        for s, b, w, v in recs:
            btwn_groups.setdefault(b, []).append(v)
        between = _one_way_anova([(k, v) for k, vs in btwn_groups.items() for v in vs])
        # 组内效应（忽略组）
        within_groups = OrderedDict()
        for s, b, w, v in recs:
            within_groups.setdefault(w, []).append(v)
        within = _one_way_anova([(k, v) for k, vs in within_groups.items() for v in vs])
        # 交互：within×between 双因素
        triples = [(b, w, v) for s, b, w, v in recs]
        interaction = _two_way_anova(triples, "III")
        return {"model": "Mixed ANOVA (within × between, subject as random)",
                "between": between, "within": within, "interaction": interaction,
                "n_subjects": len(subs), "between_levels": bs, "within_levels": ws}
    except Exception as e:
        return {"error": "混合 ANOVA 失败: %s" % e}


def _mediation(xcol, mcol, ycol, rows):
    """中介分析（Baron-Kenny + Sobel 检验）。x→m→y。"""
    try:
        xs, ms, ys = [], [], []
        for r in rows:
            x = to_float(r.get(xcol)); m = to_float(r.get(mcol)); y = to_float(r.get(ycol))
            if x is not None and m is not None and y is not None:
                xs.append(x); ms.append(m); ys.append(y)
        n = len(xs)
        if n < 4:
            return {"error": "中介分析样本不足"}
        b1 = _ols_slope(xs, ms)     # x→m
        b2 = _ols_slope(xs, ys)     # x→y 总效应 c
        b3 = _ols_slope(ms, ys)     # m→y
        # 控制 m 后 x→y 直接效应 c'
        X = [[1.0, xs[i], ms[i]] for i in range(n)]
        beta = gaussian_solve([[sum(r[a] * r[b] for r in X) for b in range(3)] for a in range(3)],
                              [sum(r[a] * ys[i] for i, r in enumerate(X)) for a in range(3)])
        c_prime = beta[1]
        indirect = b1 * b3
        total = b2
        sobel_se = math.sqrt(b3 ** 2 * _se_x(xs, ms) ** 2 + b1 ** 2 * _se_x(ms, ys) ** 2)
        z = indirect / sobel_se if sobel_se > 0 else 0.0
        p = _norm_sf(abs(z)) * 2.0
        return {"model": "mediation (x=%s, m=%s, y=%s)" % (xcol, mcol, ycol),
                "path_x_m": round(b1, 4), "path_x_y_total": round(total, 4),
                "path_m_y": round(b3, 4), "path_x_y_direct": round(c_prime, 4),
                "indirect_effect": round(indirect, 4),
                "proportion_mediated": round(indirect / total, 4) if total != 0 else None,
                "sobel_z": round(z, 4), "sobel_p": _r4(p), "significant": _sig(p)}
    except Exception as e:
        return {"error": "中介分析失败: %s" % e}


def _ols_slope(x, y):
    mx, my = mean(x), mean(y)
    denom = sum((v - mx) ** 2 for v in x)
    return sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / denom if denom > 0 else 0.0


def _se_x(x, y):
    mx, my = mean(x), mean(y)
    n = len(x)
    b = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / sum((v - mx) ** 2 for v in x)
    resid = [y[i] - (my + b * (x[i] - mx)) for i in range(n)]
    return math.sqrt(sum(r ** 2 for r in resid) / (n - 2)) / math.sqrt(sum((v - mx) ** 2 for v in x))


def _moderation(rows, spec):
    """调节分析：Y = b0 + b1*X + b2*W + b3*X*W（中心化），交互项 b3 检验 + 简单斜率 + Johnson-Neyman。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        return {"error": "moderation 需 'x,w,y'"}
    xcol, wcol, ycol = parts
    data = [(to_float(r.get(xcol)), to_float(r.get(wcol)), to_float(r.get(ycol))) for r in rows]
    data = [(x, w, y) for x, w, y in data if x is not None and w is not None and y is not None]
    n = len(data)
    if n < 6:
        return {"error": "调节分析样本不足（需 ≥6）"}
    xs = [d[0] for d in data]
    ws = [d[1] for d in data]
    ys = [d[2] for d in data]
    mx, mw = mean(xs), mean(ws)
    xc = [x - mx for x in xs]
    wc = [w - mw for w in ws]
    p = 4
    Xrows = [[1.0, xc[i], wc[i], xc[i] * wc[i]] for i in range(n)]
    rc = _regress_core(Xrows, ys, n, p, 0)
    if rc is None:
        return {"error": "回归矩阵奇异，无法拟合"}
    beta, resid, XtX_inv, cov_sel, cov_hc3 = rc
    sse = sum(r * r for r in resid)
    sst = sum((y - mean(ys)) ** 2 for y in ys)
    r2 = 1 - sse / sst if sst > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else r2
    df_resid = n - p
    ses = [math.sqrt(max(0, cov_sel[i][i])) for i in range(p)]
    t_vals = [beta[i] / ses[i] if ses[i] > 0 else 0 for i in range(p)]
    p_vals = [_t_two_sided_p(abs(t_vals[i]), df_resid) for i in range(p)]
    sd_w = stdev(ws)
    b3, p_interact = beta[3], p_vals[3]
    a = _CFG["alpha"]
    simple_slopes = []
    for label, w_val in [("low_-1SD", mw - sd_w), ("mean", mw), ("high_+1SD", mw + sd_w)]:
        wc_val = w_val - mw
        slope = beta[1] + beta[3] * wc_val
        var_slope = cov_sel[1][1] + wc_val ** 2 * cov_sel[3][3] + 2 * wc_val * cov_sel[1][3]
        se_slope = math.sqrt(max(0, var_slope))
        t_slope = slope / se_slope if se_slope > 0 else 0
        p_slope = _t_two_sided_p(abs(t_slope), df_resid)
        simple_slopes.append({"w_level": label, "w_value": round(w_val, 4),
                              "simple_slope": round(slope, 4), "se": round(se_slope, 4),
                              "t": round(t_slope, 4), "p_value": _r4(p_slope)})
    jn_points = []
    try:
        z_crit = _norm_ppf(1 - a / 2)
        if z_crit is not None:
            A = cov_sel[3][3]
            B = 2 * cov_sel[1][3]
            C = cov_sel[1][1] - z_crit ** 2
            disc = B ** 2 - 4 * A * C
            if A > 0 and disc >= 0:
                for sign in (1, -1):
                    root = (-B + sign * math.sqrt(disc)) / (2 * A)
                    jn_w = root + mw
                    jn_points.append(round(jn_w, 4))
    except Exception:
        pass
    interp = ("调节分析（X=%s, W=%s, Y=%s）：交互项 b3=%.4f, p=%s, %s（α=%g）；"
              "R²=%.4f（调整 R²=%.4f）"
              % (xcol, wcol, ycol, b3, _fmt_p(p_interact),
                 "交互显著" if _sig(p_interact, a) else "交互不显著", a, r2, adj_r2))
    return {"type": "moderation", "x": xcol, "w": wcol, "y": ycol, "n": n,
            "coefficients": {"intercept": round(beta[0], 4), "b1_x": round(beta[1], 4),
                              "b2_w": round(beta[2], 4), "b3_xw": round(beta[3], 4)},
            "se": [round(s, 4) for s in ses],
            "t_values": [round(t, 4) for t in t_vals],
            "p_values": [_r4(p) for p in p_vals],
            "r_squared": round(r2, 4), "adj_r_squared": round(adj_r2, 4),
            "interaction_p": _r4(p_interact),
            "significant": bool(_sig(p_interact, a)),
            "simple_slopes": simple_slopes,
            "johnson_neyman_points": jn_points,
            "interpretation": interp}


def _bootstrap_mediation(rows, spec, n_boot=5000, seed=0):
    """Bootstrap 中介检验（Hayes PROCESS 式）：BCa CI for indirect effect a*b。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        return {"error": "bootstrap-mediation 需 'x,m,y'"}
    xcol, mcol, ycol = parts
    base = _mediation(xcol, mcol, ycol, rows)
    if "error" in base:
        return base
    data = [(to_float(r.get(xcol)), to_float(r.get(mcol)), to_float(r.get(ycol))) for r in rows]
    data = [(x, m, y) for x, m, y in data if x is not None and m is not None and y is not None]
    n = len(data)
    if n < 6:
        return {"error": "Bootstrap 中介分析样本不足"}
    rng = random.Random(seed)
    point_indirect = base["indirect_effect"]
    indirects = []
    for bi in range(n_boot):
        if bi % 500 == 0 and bi > 0:
            _vlog("  bootstrap mediation: %d/%d" % (bi, n_boot))
        sample = [data[rng.randrange(n)] for _ in range(n)]
        sx = [d[0] for d in sample]
        sm = [d[1] for d in sample]
        sy = [d[2] for d in sample]
        a_path = _ols_slope(sx, sm)
        b_path = _ols_slope(sm, sy)
        indirects.append(a_path * b_path)
    indirects_sorted = sorted(indirects)
    lo_pct = quantile(indirects_sorted, 0.025)
    hi_pct = quantile(indirects_sorted, 0.975)
    ci_method = "percentile"
    ci_lo, ci_hi = lo_pct, hi_pct
    if n <= 3000 and n_boot >= 500:
        try:
            jk = []
            for j in range(n):
                sub = [data[i] for i in range(n) if i != j]
                sx = [d[0] for d in sub]; sm = [d[1] for d in sub]; sy = [d[2] for d in sub]
                jk.append(_ols_slope(sx, sm) * _ols_slope(sm, sy))
            jbar = mean(jk)
            num = sum((jbar - v) ** 3 for v in jk)
            den = sum((jbar - v) ** 2 for v in jk)
            if den > 0:
                acc = num / (6.0 * (den ** 1.5))
                below = sum(1 for s in indirects if s < point_indirect)
                p0 = below / len(indirects)
                if 0 < p0 < 1:
                    z0 = _norm_ppf(p0)
                    zl = _norm_ppf(0.025)
                    zh = _norm_ppf(0.975)
                    if z0 is not None and zl is not None and zh is not None:
                        def _adj(z):
                            d = 1.0 - acc * (z0 + z)
                            return _norm_cdf(z0 + (z0 + z) / d) if d > 0 else None
                        al = _adj(zl)
                        ah = _adj(zh)
                        if al and ah and 0 < al < 1 and 0 < ah < 1:
                            ci_lo = quantile(indirects_sorted, al)
                            ci_hi = quantile(indirects_sorted, ah)
                            ci_method = "bca"
        except Exception:
            pass
    sig = not (ci_lo <= 0 <= ci_hi)
    a = _CFG["alpha"]
    interp = ("Bootstrap 中介检验（n_boot=%d, seed=%d）：间接效应 a*b=%.4f, %s 95%% CI=[%.4f, %.4f], "
              "0 %s 在 CI 内，%s（α=%g）"
              % (n_boot, seed, point_indirect, ci_method, ci_lo, ci_hi,
                 "不" if sig else "", "中介显著" if sig else "中介不显著", a))
    return {"type": "bootstrap_mediation", "x": xcol, "m": mcol, "y": ycol, "n": n,
            "n_boot": n_boot, "seed": seed,
            "point_estimate": round(point_indirect, 4),
            "indirect_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "ci_method": ci_method,
            "sobel_z": base.get("sobel_z"), "sobel_p": base.get("sobel_p"),
            "proportion_mediated": base.get("proportion_mediated"),
            "significant": sig,
            "interpretation": interp}


def _stepwise(rows, spec, direction="forward", threshold=0.05):
    """逐步回归：forward / backward / best-subset。spec='y,x1,x2,...'。"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) < 3:
        return {"error": "stepwise 需 'y,x1,x2,...'（至少 2 个自变量）"}
    ycol = parts[0]
    xcols = parts[1:]
    data = []
    for r in rows:
        y = to_float(r.get(ycol))
        if y is None:
            continue
        xs = [to_float(r.get(c)) for c in xcols]
        if all(x is not None for x in xs):
            data.append((xs, y))
    n = len(data)
    k = len(xcols)
    if n < k + 2:
        return {"error": "样本量不足（n=%d < k+2=%d）" % (n, k + 2)}
    Y = [d[1] for d in data]
    all_X = [d[0] for d in data]
    sst = sum((y - mean(Y)) ** 2 for y in Y)
    def _fit_subset(idx_list):
        p = len(idx_list) + 1
        Xrows = [[1.0] + [all_X[i][j] for j in idx_list] for i in range(n)]
        rc = _regress_core(Xrows, Y, n, p, 0)
        if rc is None:
            return None
        beta, resid = rc[0], rc[1]
        sse = sum(r * r for r in resid)
        r2 = 1 - sse / sst if sst > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else r2
        aic = n * math.log(sse / n) + 2 * p if sse > 0 else float("inf")
        bic = n * math.log(sse / n) + p * math.log(n) if sse > 0 else float("inf")
        cov = rc[3]
        ses = [math.sqrt(max(0, cov[i][i])) for i in range(p)]
        t_vals = [beta[i] / ses[i] if ses[i] > 0 else 0 for i in range(p)]
        p_vals = [_t_two_sided_p(abs(t_vals[i]), n - p) for i in range(p)]
        return {"beta": beta, "sse": sse, "r2": r2, "adj_r2": adj_r2,
                "aic": aic, "bic": bic, "p_vals": p_vals, "ses": ses, "t_vals": t_vals}
    steps = []
    if direction == "best_subset":
        from itertools import combinations
        best = None
        best_combo = None
        for r in range(1, k + 1):
            for combo in combinations(range(k), r):
                fit = _fit_subset(list(combo))
                if fit is None:
                    continue
                if best is None or fit["adj_r2"] > best["adj_r2"]:
                    best = fit
                    best_combo = list(combo)
        if best is None:
            return {"error": "所有子集均无法拟合"}
        steps.append({"step": 1, "action": "best_subset",
                      "selected": [xcols[j] for j in best_combo],
                      "adj_r2": round(best["adj_r2"], 4), "aic": round(best["aic"], 2),
                      "bic": round(best["bic"], 2)})
        final_fit = best
        final_cols = best_combo
    elif direction == "forward":
        selected = []
        remaining = list(range(k))
        step_num = 0
        while remaining:
            best_j = None
            best_p = None
            best_fit = None
            for j in remaining:
                fit = _fit_subset(selected + [j])
                if fit is None:
                    continue
                p_val = fit["p_vals"][-1]
                if best_p is None or p_val < best_p:
                    best_p = p_val
                    best_j = j
                    best_fit = fit
            if best_j is None or best_p is None or best_p >= threshold:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            step_num += 1
            steps.append({"step": step_num, "action": "add %s" % xcols[best_j],
                          "p_value": _r4(best_p), "adj_r2": round(best_fit["adj_r2"], 4)})
        final_fit = best_fit
        final_cols = selected
    else:
        selected = list(range(k))
        while len(selected) > 1:
            fit = _fit_subset(selected)
            if fit is None:
                break
            worst_p = 0
            worst_idx = -1
            for i, j in enumerate(selected):
                p_val = fit["p_vals"][i + 1]
                if p_val > worst_p:
                    worst_p = p_val
                    worst_idx = i
            if worst_p < threshold:
                break
            removed = selected.pop(worst_idx)
            steps.append({"step": len(steps) + 1, "action": "remove %s" % xcols[removed],
                          "p_value": _r4(worst_p), "adj_r2": round(fit["adj_r2"], 4)})
        final_fit = _fit_subset(selected)
        final_cols = selected
    if not final_cols or final_fit is None:
        return {"error": "逐步回归未能选出有效模型"}
    a = _CFG["alpha"]
    interp = ("逐步回归（%s, threshold=%g）：最终模型含 %d 个自变量 %s，"
              "R²=%.4f, 调整 R²=%.4f, AIC=%.2f, BIC=%.2f"
              % (direction, threshold, len(final_cols),
                 ", ".join(xcols[j] for j in final_cols),
                 final_fit["r2"], final_fit["adj_r2"],
                 final_fit["aic"], final_fit["bic"]))
    coefs = {}
    for i, j in enumerate(final_cols):
        coefs[xcols[j]] = {"beta": round(final_fit["beta"][i + 1], 4),
                           "se": round(final_fit["ses"][i + 1], 4),
                           "t": round(final_fit["t_vals"][i + 1], 4),
                           "p_value": _r4(final_fit["p_vals"][i + 1])}
    return {"type": "stepwise", "direction": direction, "y": ycol,
            "selected_predictors": [xcols[j] for j in final_cols],
            "coefficients": coefs,
            "intercept": round(final_fit["beta"][0], 4),
            "r_squared": round(final_fit["r2"], 4),
            "adj_r_squared": round(final_fit["adj_r2"], 4),
            "aic": round(final_fit["aic"], 2), "bic": round(final_fit["bic"], 2),
            "steps": steps,
            "interpretation": interp}


# ======================================================================
#  T3⑥ 结构方程模型（路径分析 + 轻量 CFA）与项目反应理论（IRT）
#  零依赖实现，复用 _mat_inv / _det_sym / _regress_core / _sigmoid
# ======================================================================

def _jacobi_eigh(M):
    """对称矩阵 Jacobi 特征分解（零依赖），返回 (特征值降序列表, 特征向量列表)。
    特征向量 vec[i] 为第 i 个特征向量（list）。用于 Gauss-Legendre 求积权重。
    注意：旋转时只更新 i≠p,q 的非对角项，轴元素用原始 app/aqq/apq 计算，
    避免把已更新元素再次参与计算导致特征值失真。"""
    n = len(M)
    if n == 0:
        return [], []
    if n == 1:
        return [float(M[0][0])], [[1.0]]
    A = [list(r) for r in M]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(200):
        p, q, maxv = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > abs(maxv):
                    maxv = A[i][j]; p, q = i, j
        if abs(maxv) < 1e-12:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        # 旋转角须与更新所用的 R=[[c,s],[-s,c]] 自洽：令 b''=(aqq-app)/2·sin2φ+apq·cos2φ=0
        # ⇒ tan(2φ)=2·apq/(aqq-app)。原分母 (app-aqq) 符号反致 b'' 未消却强行置 0，
        # 得非相似矩阵、特征值失真（仅 a==d 特例正确），污染 SEM/CFA 主轴因子法。
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for i in range(n):
            if i != p and i != q:
                aip, aiq = A[i][p], A[i][q]
                A[i][p] = c * aip - s * aiq
                A[i][q] = s * aip + c * aiq
                A[p][i] = A[i][p]
                A[q][i] = A[i][q]
        A[p][p] = c * c * app + s * s * aqq - 2.0 * c * s * apq
        A[q][q] = s * s * app + c * c * aqq + 2.0 * c * s * apq
        A[p][q] = 0.0
        A[q][p] = 0.0
        for i in range(n):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq
    eig = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -eig[i])
    eig_sorted = [eig[i] for i in order]
    vec = [[V[r][order[pos]] for r in range(n)] for pos in range(n)]
    return eig_sorted, vec


def _tridiag_eig_bisect(d, e):
    """对称三对角矩阵（对角 d，次对角 e，长 n-1）特征值，Sturm 二分法。
    用于 Gauss 求积节点生成。原 _jacobi_eigh 在零对角三对角阵上不收敛（off-diagonal
    范数停滞），而 Gauss-Legendre / Gauss-Hermite 的 Jacobi 矩阵恰为零对角，故用本函数
    取代，规避该数值缺陷。返回特征值升序列表。"""
    n = len(d)

    def sign_changes(lam):
        pprev, pcur = 0.0, 1.0  # p_{-1}=0, p_0=1
        sc = 0
        for k in range(1, n + 1):
            pk = (lam - d[k - 1]) * pcur - ((e[k - 2] ** 2) * pprev if k >= 2 else 0.0)
            if pk == 0:
                pk = 1e-300
            if pcur * pk < 0:
                sc += 1
            pprev, pcur = pcur, pk
        return sc

    def count_le(lam):
        return n - sign_changes(lam)

    # Gershgorin 区间（e_{-1}=e_{n-1}=0），保证包含全部特征值。朴素下界 d[0]-|e[0]|-1
    # 仅计入首个次对角，对宽谱（如 Hermite，最小特征值远超出该区间）会漏掉最负特征值。
    def _radius(k):
        ep = e[k - 1] if k >= 1 else 0.0
        en = e[k] if k < n - 1 else 0.0
        return abs(ep) + abs(en)

    lo = min(d[k] - _radius(k) for k in range(n))
    hi = max(d[k] + _radius(k) for k in range(n))
    ev = []
    left = lo
    for idx in range(n):
        a, b = left, hi
        for _ in range(300):
            m = 0.5 * (a + b)
            if count_le(m) < idx + 1:
                a = m
            else:
                b = m
        val = 0.5 * (a + b)
        ev.append(val)
        left = val
    return ev




def _gauss_hermite(n):
    """Gauss-Hermite 节点/权重，专用于 IRT 能力积分 ∫ g(θ)φ(θ)dθ（φ 为标准正态）。
    节点 θ_i = √2·x_i（x_i 为物理 Hermite 多项式 H_n 之根）；权重已归一化（总质量=1），
    即 N(0,1) 积分权重 W_i^φ = 2^{n-1}·n! / (n^2·H_{n-1}(x_i)^2)。
    稳定计算：系数取对数空间（math.lgamma）避免大 n 时 2^{n-1}·n! 溢出；H_{n-1} 由递推求值。
    取代原依赖 _jacobi_eigh（零对角阵上不收敛）的实现，2PL 区分度估计不再被低估/高估。"""
    d = [0.0] * n
    e = [math.sqrt((k + 1) / 2.0) for k in range(n - 1)]
    xs = _tridiag_eig_bisect(d, e)  # Hermite 节点 x（权 e^{-x^2}）
    # ln(W_i^φ) = (n-1)·ln2 + ln(n!) − 2·ln n − 2·ln|H_{n-1}(x_i)|
    ln_coef = (n - 1) * math.log(2.0) + math.lgamma(n + 1) - 2.0 * math.log(float(n))
    wts = []
    for x in xs:
        h0, h1 = 1.0, 2.0 * x
        for k in range(1, n):
            h2 = 2.0 * x * h1 - 2.0 * k * h0
            h0, h1 = h1, h2
        Hnm1 = h0  # H_{n-1}(x)
        if Hnm1 == 0.0:
            Hnm1 = 1e-300
        log_w = ln_coef - 2.0 * math.log(abs(Hnm1))
        wts.append(math.exp(log_w))
    nodes = [math.sqrt(2.0) * x for x in xs]
    return nodes, wts


def _sem(model, columns, rows, alpha=None):
    """结构方程模型 / 路径分析（观测变量 + 轻量 CFA 潜变量），含拟合指数。

    model 方程串（分号分隔）：
      'y ~ x1 + x2'          观测变量 y 对其预测变量（观测或潜变量）回归
      'F := i1 + i2 + i3'    潜变量 F 由其指标 i1..i3 测量（CFA）

    估计：方程逐条 OLS；潜变量用标准化合成因子分代理，首载荷固定=1 识别；
    由样本协方差 S 与模型隐含协方差 Σ 的 ML 差异函数 F 计算 χ²/CFI/TLI/RMSEA/SRMR。
    零依赖。"""
    n = len(rows)
    a = alpha or _CFG["alpha"]
    eqs = [e.strip() for e in model.split(";") if e.strip()]
    if not eqs:
        return {"error": "sem 模型为空（示例：'y ~ x1 + x2; F := i1 + i2 + i3'）"}
    regress = {}
    latent_defs = {}
    for e in eqs:
        if ":=" in e:
            lhs, rhs = e.split(":=")
            lhs = lhs.strip()
            inds = [x.strip() for x in rhs.split("+") if x.strip()]
            if len(inds) < 2:
                return {"error": "潜变量 %s 至少需要 2 个指标" % lhs}
            if lhs in latent_defs:
                return {"error": "潜变量 %s 重复定义" % lhs}
            latent_defs[lhs] = inds
            regress.setdefault(lhs, [])
        elif "~" in e:
            lhs, rhs = e.split("~")
            lhs = lhs.strip()
            preds = [x.strip() for x in rhs.split("+") if x.strip()]
            if not preds:
                return {"error": "方程 %s 缺少预测变量" % lhs}
            regress[lhs] = preds
        else:
            return {"error": "无法解析方程（需含 ~ 或 :=）：%s" % e}
    colset = set(columns)
    referenced = set()
    for lhs, preds in regress.items():
        referenced.add(lhs)
        for p in preds:
            referenced.add(p)
    for f, inds in latent_defs.items():
        for i in inds:
            referenced.add(i)
    for v in referenced:
        if v in colset or v in latent_defs:
            continue
        return {"error": "变量 %s 既非数据列也非已定义潜变量" % v}
    # 潜变量：单公因子主轴因子法（PAF）。标准化指标→相关矩阵→在约化相关矩阵
    # （对角=共同度）上迭代取第一特征向量，使载荷匹配非对角结构（比 PCA 更准）；
    # 因子分取第一主成分得分并标准化为单位方差，保证与 var(F)=1 的载荷尺度一致。
    factor_score = {}
    meas = {}
    for f, inds in latent_defs.items():
        k = len(inds)
        stat = {}
        for i in inds:
            col = [to_float(r.get(i)) for r in rows]
            col = [x for x in col if x is not None]
            stat[i] = (mean(col) if col else 0.0, stdev(col) if len(col) > 1 else 0.0)
        z = {}
        for i in inds:
            m, s = stat[i]
            z[i] = [(to_float(r.get(i)) - m) / s if (s > 0 and to_float(r.get(i)) is not None) else None
                    for r in rows]
        R = [[0.0] * k for _ in range(k)]
        for a in range(k):
            for b in range(a, k):
                ca, cb = inds[a], inds[b]
                pairs = [(z[ca][t], z[cb][t]) for t in range(n)
                         if z[ca][t] is not None and z[cb][t] is not None]
                if len(pairs) < 2:
                    R[a][b] = R[b][a] = (1.0 if a == b else 0.0)
                    continue
                aa = [p[0] for p in pairs]; bb = [p[1] for p in pairs]
                ma, mb = mean(aa), mean(bb)
                va = sum((x - ma) ** 2 for x in aa) / len(aa)
                vb = sum((x - mb) ** 2 for x in bb) / len(aa)
                if va <= 0 or vb <= 0:
                    R[a][b] = R[b][a] = (1.0 if a == b else 0.0)
                    continue
                r = sum((aa[t] - ma) * (bb[t] - mb) for t in range(len(aa))) / (len(aa) * math.sqrt(va * vb))
                R[a][b] = R[b][a] = r
        # 主轴因子法迭代
        h2 = [0.5] * k
        l1 = 1.0
        e1 = [1.0 / math.sqrt(k)] * k
        for _ in range(100):
            Rstar = [[R[a][b] if a != b else h2[a] for b in range(k)] for a in range(k)]
            eig, vec = _jacobi_eigh(Rstar)
            if not eig:
                break
            l1 = eig[0]
            e1 = vec[0] if vec else e1
            new_h2 = [(e1[a] * math.sqrt(max(l1, 0.0))) ** 2 for a in range(k)]
            diff = max(abs(new_h2[a] - h2[a]) for a in range(k))
            h2 = new_h2
            if diff < 1e-7:
                break
        l_std = [e1[a] * math.sqrt(max(l1, 0.0)) for a in range(k)]
        # 第一主成分得分（因子分代理）
        comps_raw = []
        for t in range(n):
            ssum = 0.0
            ok = True
            for a, i in enumerate(inds):
                zv = z[i][t]
                if zv is None:
                    ok = False
                    break
                ssum += e1[a] * zv
            comps_raw.append(ssum if ok else None)
        # 标准化为单位方差，使结构模型中的 var(F)=1 与载荷尺度一致
        vals = [c for c in comps_raw if c is not None]
        fm = mean(vals) if vals else 0.0
        fs = stdev(vals) if len(vals) > 1 else 0.0
        if fs > 0:
            factor_score[f] = [(c - fm) / fs if c is not None else None for c in comps_raw]
        else:
            factor_score[f] = [(c - fm) if c is not None else None for c in comps_raw]
        m = {}
        for a, i in enumerate(inds):
            mm, ss = stat[i]
            lam = l_std[a] * ss
            comm = l_std[a] * l_std[a]
            u = max(1.0 - comm, 1e-3)
            psi = u * ss * ss
            m[i] = (lam, psi)
        meas[f] = m

    def _vals(v):
        if v in factor_score:
            return factor_score[v]
        return [to_float(r.get(v)) for r in rows]

    # 回归方程估计
    est = {}
    for lhs, preds in regress.items():
        if not preds:
            continue
        Yv = _vals(lhs)
        Xdat = [_vals(p) for p in preds]
        data = []
        for t in range(n):
            yv = Yv[t]
            xs = [Xdat[j][t] for j in range(len(preds))]
            if yv is None or any(x is None for x in xs):
                continue
            data.append([yv] + xs)
        if len(data) < len(preds) + 2:
            return {"error": "方程 %s 有效样本不足（n=%d）" % (lhs, len(data))}
        Yd = [d[0] for d in data]
        Xd = [[1.0] + [d[j + 1] for j in range(len(preds))] for d in data]
        rc = _regress_core(Xd, Yd, len(data), len(preds) + 1, 0)
        if rc is None:
            return {"error": "方程 %s 设计矩阵奇异" % lhs}
        beta, resid = rc[0], rc[1]
        sse = sum(r * r for r in resid)
        sst = sum((y - mean(Yd)) ** 2 for y in Yd)
        rv = sse / len(data)
        r2 = 1 - sse / sst if sst > 0 else 0.0
        est[lhs] = {"slopes": {preds[j]: beta[j + 1] for j in range(len(preds))},
                    "resid": rv, "r2": r2, "n": len(data)}
    # 依赖与拓扑
    def factor_of(v):
        for f, inds in latent_defs.items():
            if v in inds:
                return f
        return None

    def deps(v):
        d = []
        if v in regress and regress[v]:
            d = list(regress[v])
        f = factor_of(v)
        if f is not None:
            d = d + [f]
        return d


    nodes = list(referenced)
    order = []
    state = {}
    cycle = [False]

    def visit(v):
        st = state.get(v, 0)
        if st == 2:
            return
        if st == 1:
            cycle[0] = True
            return
        state[v] = 1
        for d in deps(v):
            if d in referenced:
                visit(d)
        state[v] = 2
        order.append(v)

    for v in nodes:
        visit(v)
    if cycle[0]:
        return {"error": "模型存在循环依赖（如 a ~ b 且 b ~ a）"}
    # 样本协方差缓存
    _scache = {}

    def sample_cov(v, w):
        if v == w:
            A = _vals(v)
            arr = [x for x in A if x is not None]
            if len(arr) < 2:
                return 0.0
            mm = mean(arr)
            return sum((x - mm) ** 2 for x in arr) / len(arr)
        key = (v, w) if v < w else (w, v)
        if key in _scache:
            return _scache[key]
        A, B = _vals(v), _vals(w)
        d = [(A[t], B[t]) for t in range(n) if A[t] is not None and B[t] is not None]
        if len(d) < 2:
            _scache[key] = 0.0
            return 0.0
        aa = [x[0] for x in d]; bb = [x[1] for x in d]
        ma, mb = mean(aa), mean(bb)
        c = sum((aa[t] - ma) * (bb[t] - mb) for t in range(len(aa))) / len(aa)
        _scache[key] = c
        return c

    obs_vars = [v for v in referenced if v in colset]
    p = len(obs_vars)
    if p < 2:
        return {"error": "sem 需至少 2 个观测变量参与模型拟合"}
    # ---- 结构矩阵法构建隐含协方差 Σ（保证对称一致，避免递归法的 reduced-form 不自洽）----
    allvars = list(referenced)
    m = len(allvars)
    def _is_ind(v):
        return factor_of(v) is not None
    M = [v for v in allvars if (v in regress and regress[v]) or _is_ind(v)]
    E = [v for v in allvars if v not in M]
    m_e, m_x = len(M), len(E)
    idxM = {v: i for i, v in enumerate(M)}
    idxE = {v: i for i, v in enumerate(E)}
    B = [[0.0] * m_e for _ in range(m_e)]
    Gamma = [[0.0] * m_x for _ in range(m_e)]
    Psi = [[0.0] * m_e for _ in range(m_e)]
    for vi, v in enumerate(M):
        if v in regress and regress[v]:
            sl = est[v]["slopes"]; rv = est[v]["resid"]
            for pp, beta in sl.items():
                if pp in idxM:
                    B[vi][idxM[pp]] = beta
                else:
                    Gamma[vi][idxE[pp]] = beta
            Psi[vi][vi] = rv
        else:
            f = factor_of(v); lam, rv = meas[f][v]
            if f in idxM:
                B[vi][idxM[f]] = lam
            else:
                Gamma[vi][idxE[f]] = lam
            Psi[vi][vi] = rv
    Phi = [[0.0] * m_x for _ in range(m_x)]
    for i, a in enumerate(E):
        for j, b in enumerate(E):
            if a in latent_defs or b in latent_defs:
                # 潜变量方差固定=1（PCA 载荷已含该尺度），不同潜变量假定正交
                Phi[i][j] = 1.0 if a == b else 0.0
            else:
                Phi[i][j] = sample_cov(a, b)
    IB = [[(1.0 if i == j else 0.0) - B[i][j] for j in range(m_e)] for i in range(m_e)]
    IB_inv = _mat_inv(IB)
    if IB_inv is None:
        return {"error": "结构矩阵 (I-B) 奇异，模型可能未识别"}
    GP = _mat_mul(Gamma, Phi)
    GPGt = _mat_mul(GP, _mat_transpose(Gamma))
    Mid = [[GPGt[i][j] + Psi[i][j] for j in range(m_e)] for i in range(m_e)]
    Sig_eta = _mat_mul(_mat_mul(IB_inv, Mid), _mat_transpose(IB_inv))
    Sig_eta_xi = _mat_mul(IB_inv, _mat_mul(Gamma, Phi))
    Sig_all = [[0.0] * m for _ in range(m)]
    for i in range(m_e):
        for j in range(m_e):
            Sig_all[i][j] = Sig_eta[i][j]
        for j in range(m_x):
            Sig_all[i][m_e + j] = Sig_eta_xi[i][j]
            Sig_all[m_e + j][i] = Sig_eta_xi[i][j]
    for i in range(m_x):
        for j in range(m_x):
            Sig_all[m_e + i][m_e + j] = Phi[i][j]
    order_idx = M + E
    obs_pos = [order_idx.index(v) for v in obs_vars]
    S = [[sample_cov(v, w) for w in obs_vars] for v in obs_vars]
    Sig = [[Sig_all[r][c] for c in obs_pos] for r in obs_pos]
    Sdet = _det_sym(S); Sigdet = _det_sym(Sig)
    if Sdet is None or Sdet <= 0 or Sigdet is None or Sigdet <= 0:
        return {"error": "协方差矩阵奇异，无法计算拟合指数"}
    Sig_inv = _mat_inv(Sig)
    if Sig_inv is None:
        return {"error": "隐含协方差矩阵不可逆"}
    tr = sum(S[i][j] * Sig_inv[i][j] for i in range(p) for j in range(p))
    Fml = math.log(Sigdet) + tr - math.log(Sdet) - p
    if Fml < 0:
        Fml = 0.0
    chi2 = (n - 1) * Fml
    q = 0
    for lhs, preds in regress.items():
        if preds:
            q += len(preds) + 1
    for f, inds in latent_defs.items():
        q += (len(inds) - 1) + len(inds)
        if not (f in regress and regress[f]):
            q += 1
    df = p * (p + 1) // 2 - q
    if df < 1:
        return {"error": "模型自由度不足（df=%d），参数过多可能未识别" % df}
    chi2_p = _chi2_sf(chi2, df)
    logdetS = math.log(Sdet)
    diag_sum = sum(math.log(S[i][i]) for i in range(p) if S[i][i] > 0)
    F_null = max(logdetS - diag_sum, 0.0)
    chi2_null = (n - 1) * F_null
    df_null = p * (p - 1) // 2
    cfi = None
    if chi2_null - df_null > 0:
        cfi = max(0.0, min(1.0, 1.0 - (chi2 - df) / (chi2_null - df_null)))
    tli = None
    if df > 0 and df_null > 0 and (chi2_null / df_null - 1) != 0:
        tli = max(0.0, min(1.0, (chi2_null / df_null - chi2 / df) / (chi2_null / df_null - 1)))
    rmsea = math.sqrt(max(chi2 - df, 0.0) / (df * (n - 1))) if df > 0 else None
    num = 0.0
    for i in range(p):
        for j in range(i + 1):
            d = S[i][j] - Sig[i][j]
            num += d * d
    srmr = math.sqrt(2.0 * num / (p * (p + 1)))
    reg_out = {}
    for lhs, preds in regress.items():
        if not preds:
            continue
        reg_out[lhs] = {"predictors": {p_: round(est[lhs]["slopes"][p_], 4) for p_ in preds},
                        "residual_var": round(est[lhs]["resid"], 4),
                        "r_squared": round(est[lhs]["r2"], 4), "n": est[lhs]["n"]}
    meas_out = {}
    for f, inds in latent_defs.items():
        meas_out[f] = {"indicators": {i: {"loading": round(meas[f][i][0], 4),
                                          "residual_var": round(meas[f][i][1], 4)} for i in inds}}
    interp = ("结构方程模型（路径分析%s）：χ²(%.0f)=%.3f, p=%s；CFI=%.3f, TLI=%.3f, "
              "RMSEA=%.3f, SRMR=%.3f（观测变量 %d 个，自由参数 %d，df=%d）"
              % (" + CFA 潜变量" if latent_defs else "",
                 df, chi2, _fmt_p(chi2_p), cfi if cfi is not None else float("nan"),
                 tli if tli is not None else float("nan"),
                 rmsea if rmsea is not None else float("nan"), srmr, p, q, df))
    return {"type": "sem", "model": model, "n": n, "observed_vars": p,
            "fit": {"chi_square": round(chi2, 4), "df": df, "p_value": _r4(chi2_p),
                    "cfi": _r4(cfi), "tli": _r4(tli), "rmsea": _r4(rmsea), "srmr": round(srmr, 4)},
            "equations": {lhs: (regress[lhs] or ["(latent)"]) for lhs in regress},
            "regression": reg_out, "measurement": meas_out,
            "interpretation": interp,
            "method_note": "观测变量用 OLS 逐方程估计；潜变量以标准化合成因子分代理并固定首载荷=1 识别；"
                           "拟合指数由样本协方差与模型隐含协方差的 ML 差异函数 F 计算（零依赖）。"}


def cmd_sem(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    model = args.model
    if args.model_file:
        try:
            with open(args.model_file, "r", encoding="utf-8") as f:
                model = f.read()
        except Exception as e:
            die("模型文件读取失败: %s" % e)
    if not model:
        die("sem 需 --model '方程串' 或 --model-file 路径")
    res = _sem(model, columns, rows, float(args.alpha) if args.alpha else None)
    if "error" in res:
        die(res["error"])
    emit({"status": "ok", "task": "sem", "result": res})


def _irt_dich_prob(aj, bj, cj, theta):
    p = _sigmoid(aj * (theta - bj))
    return cj + (1.0 - cj) * p


def _irt_poly_prob(aj, thr_j, model_type, theta, K):
    """分类计分题在能力 θ 下的各类别概率。GRM：累积 logit 差分；PCM：Andrich 部分计分。"""
    if model_type == "grm":
        Gc = [1.0]
        for m in range(1, K + 1):
            Gc.append(_sigmoid(aj * (theta - thr_j[m - 1])))
        pk = []
        for k in range(K + 1):
            gk = Gc[k]
            gk1 = Gc[k + 1] if k + 1 <= K else 0.0
            pk.append(max(gk - gk1, 1e-300))
        return pk
    else:  # pcm
        T = [0.0]
        for k in range(1, K + 1):
            T.append(T[-1] + (theta - thr_j[k - 1]))
        mx = max(T)
        exps = [math.exp(t - mx) for t in T]
        Z = sum(exps)
        return [e / Z for e in exps]


def _irt_itemQ_dich(aj, bj, cj, j, Ug, Ng, nodes, G):
    ugj = Ug[j]
    s = 0.0
    for g in range(G):
        P = _irt_dich_prob(aj, bj, cj, nodes[g])
        if P < 1e-12:
            P = 1e-12
        elif P > 1.0 - 1e-12:
            P = 1.0 - 1e-12
        s += ugj[g] * math.log(P) + (Ng[g] - ugj[g]) * math.log(1.0 - P)
    return s


def _irt_dich_mstep(j, a, b, c, is_3pl, Ug, Ng, nodes, G, model_type):
    """逐题 M-step：1PL 仅 b；2PL/3PL 估计 a,b（及 3PL 的猜测 c）。配步长折半保证 Q 单调不降。"""
    ugj = Ug[j]
    cj0 = (c[j] if is_3pl else 0.0)
    if model_type == "1pl":
        a[j] = 1.0
        for _ in range(100):
            gb = 0.0; ibb = 0.0
            for g in range(G):
                P = _sigmoid(a[j] * (nodes[g] - b[j])); Qp = P * (1 - P)
                diff = ugj[g] - Ng[g] * P
                gb += -a[j] * diff; ibb += Ng[g] * Qp
            if ibb <= 0:
                break
            step = gb / ibb; newb = b[j] + step
            if _irt_itemQ_dich(a[j], newb, cj0, j, Ug, Ng, nodes, G) >= _irt_itemQ_dich(a[j], b[j], cj0, j, Ug, Ng, nodes, G) - 1e-12:
                b[j] = newb
            else:
                imp = False
                for _h in range(30):
                    step *= 0.5; newb = b[j] + step
                    if _irt_itemQ_dich(a[j], newb, cj0, j, Ug, Ng, nodes, G) >= _irt_itemQ_dich(a[j], b[j], cj0, j, Ug, Ng, nodes, G) - 1e-12:
                        b[j] = newb; imp = True; break
                if not imp:
                    break
            if abs(step) < 1e-6:
                break
        if b[j] < -4.0: b[j] = -4.0
        if b[j] > 4.0: b[j] = 4.0
        return
    for _ in range(100):
        ga = 0.0; gb = 0.0; iaa = 0.0; iab = 0.0; ibb = 0.0
        cj = c[j] if is_3pl else 0.0
        for g in range(G):
            tb = nodes[g] - b[j]
            P = _irt_dich_prob(a[j], b[j], cj, nodes[g]); Qp = P * (1 - P)
            diff = ugj[g] - Ng[g] * P
            ga += diff * tb; gb += -a[j] * diff
            w = Ng[g] * Qp
            iaa += w * tb * tb; iab += -w * a[j] * tb; ibb += w * a[j] * a[j]
        det = iaa * ibb - iab * iab
        if det <= 0:
            break
        iinv_aa = ibb / det; iinv_ab = -iab / det; iinv_bb = iaa / det
        da = iinv_aa * ga + iinv_ab * gb; db = iinv_ab * ga + iinv_bb * gb
        q0 = _irt_itemQ_dich(a[j], b[j], cj, j, Ug, Ng, nodes, G)
        na, nb = a[j] + da, b[j] + db
        if _irt_itemQ_dich(na, nb, cj, j, Ug, Ng, nodes, G) >= q0 - 1e-12:
            a[j], b[j] = na, nb
        else:
            imp = False
            for _h in range(30):
                da *= 0.5; db *= 0.5; na, nb = a[j] + da, b[j] + db
                if _irt_itemQ_dich(na, nb, cj, j, Ug, Ng, nodes, G) >= q0 - 1e-12:
                    a[j], b[j] = na, nb; imp = True; break
            if not imp:
                break
        if abs(da) < 1e-6 and abs(db) < 1e-6:
            break
    if a[j] < 0.1: a[j] = 0.1
    if a[j] > 4.0: a[j] = 4.0
    if b[j] < -4.0: b[j] = -4.0
    if b[j] > 4.0: b[j] = 4.0
    if is_3pl:
        for _ci in range(30):
            cj = c[j]
            gc = 0.0; igc = 0.0
            for g in range(G):
                gv = _sigmoid(a[j] * (nodes[g] - b[j]))
                P = cj + (1.0 - cj) * gv
                if P < 1e-12: P = 1e-12
                elif P > 1.0 - 1e-12: P = 1.0 - 1e-12
                diff = ugj[g] - Ng[g] * P
                dPdc = (1.0 - gv)
                gc += diff / (P * (1.0 - P)) * dPdc
                igc += Ng[g] * (dPdc * dPdc) / (P * (1.0 - P))
            if igc <= 0:
                break
            step = gc / igc; nc = max(0.0, min(0.4, cj + step))
            q0 = _irt_itemQ_dich(a[j], b[j], cj, j, Ug, Ng, nodes, G)
            if _irt_itemQ_dich(a[j], b[j], nc, j, Ug, Ng, nodes, G) >= q0 - 1e-12:
                c[j] = nc
            else:
                imp = False
                for _h in range(30):
                    step *= 0.5; nc = max(0.0, min(0.4, cj + step))
                    if _irt_itemQ_dich(a[j], b[j], nc, j, Ug, Ng, nodes, G) >= q0 - 1e-12:
                        c[j] = nc; imp = True; break
                if not imp:
                    break
            if abs(step) < 1e-7:
                break
        if c[j] < 0.0: c[j] = 0.0
        if c[j] > 0.4: c[j] = 0.4


def _irt_itemQ_poly(aj, thr_j, model_type, Njcj, nodes, G, K):
    s = 0.0
    for g in range(G):
        d = Njcj[g]
        if not d:
            continue
        pk = _irt_poly_prob(aj, thr_j, model_type, nodes[g], K)
        for c, cnt in d.items():
            if cnt > 0 and pk[c] > 1e-300:
                s += cnt * math.log(pk[c])
    return s


def _irt_poly_mstep(j, thr, a, model_type, Njc, Ng, nodes, G, K):
    """逐题 M-step（多分类）：GRM 估计 a 与有序阈值 b_1..b_K；PCM 估计阈值 δ_1..δ_K（a 固定为 1）。"""
    n_gc = [[0.0] * (K + 1) for _ in range(G)]
    for g in range(G):
        for c, val in Njc[j][g].items():
            n_gc[g][c] = val
    if model_type == "grm":
        aj = a[j]
        for _it in range(80):
            ga = 0.0; iaa = 0.0
            for g in range(G):
                if Ng[g] <= 0:
                    continue
                pk = _irt_poly_prob(aj, thr[j], model_type, nodes[g], K)
                Gc = [1.0]
                for m in range(1, K + 1):
                    Gc.append(_sigmoid(aj * (nodes[g] - thr[j][m - 1])))
                for c in range(K + 1):
                    gc_c = Gc[c]
                    gc_c1 = Gc[c + 1] if c + 1 <= K else 0.0
                    bc = thr[j][c - 1] if c >= 1 else None
                    bc1 = thr[j][c] if (c + 1) <= K else None
                    term1 = (nodes[g] - bc) * gc_c * (1 - gc_c) if bc is not None else 0.0
                    term2 = (nodes[g] - bc1) * gc_c1 * (1 - gc_c1) if bc1 is not None else 0.0
                    dPc_da = term1 - term2
                    nc = n_gc[g][c]
                    if nc > 0 and pk[c] > 1e-12:
                        ga += nc / pk[c] * dPc_da
                        iaa += nc / (pk[c] * pk[c]) * dPc_da * dPc_da
            if iaa <= 0:
                break
            step = ga / iaa; na = aj + step
            if _irt_itemQ_poly(na, thr[j], model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(aj, thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                aj = na
            else:
                imp = False
                for _h in range(30):
                    step *= 0.5; na = aj + step
                    if _irt_itemQ_poly(na, thr[j], model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(aj, thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                        aj = na; imp = True; break
                if not imp:
                    break
            if abs(step) < 1e-6:
                break
        a[j] = max(0.1, min(4.0, aj))
        for m in range(K):
            for _it in range(80):
                gb = 0.0; ibb = 0.0
                for g in range(G):
                    if Ng[g] <= 0:
                        continue
                    pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                    Gc = [1.0]
                    for mm in range(1, K + 1):
                        Gc.append(_sigmoid(a[j] * (nodes[g] - thr[j][mm - 1])))
                    Gm = Gc[m + 1] if (m + 1) <= K else 0.0
                    dPc_db = -a[j] * Gm * (1 - Gm)
                    for c in range(K + 1):
                        nc = n_gc[g][c]
                        if nc > 0 and pk[c] > 1e-12:
                            indicator = (1.0 if c == (m + 1) else 0.0) - (1.0 if c == m else 0.0)
                            dPc = dPc_db * indicator
                            gb += nc / pk[c] * dPc
                            ibb += nc / (pk[c] * pk[c]) * dPc * dPc
                if ibb <= 0:
                    break
                step = gb / ibb
                newb = thr[j][m] + step
                newthr = [thr[j][x] if x != m else newb for x in range(K)]
                if _irt_itemQ_poly(a[j], newthr, model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(a[j], thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                    thr[j][m] = newb
                else:
                    imp = False
                    for _h in range(30):
                        step *= 0.5; newb = thr[j][m] + step
                        newthr = [thr[j][x] if x != m else newb for x in range(K)]
                        if _irt_itemQ_poly(a[j], newthr, model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(a[j], thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                            thr[j][m] = newb; imp = True; break
                    if not imp:
                        break
                if abs(step) < 1e-6:
                    break
            if thr[j][m] < -4.0: thr[j][m] = -4.0
            if thr[j][m] > 4.0: thr[j][m] = 4.0
        for m in range(1, K):
            if thr[j][m] < thr[j][m - 1]:
                thr[j][m] = thr[j][m - 1]
    else:  # pcm
        for m in range(K):
            d = m + 1
            for _it in range(80):
                gd = 0.0; idd = 0.0
                for g in range(G):
                    Ng_g = sum(n_gc[g])
                    if Ng_g <= 0:
                        continue
                    pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                    cg = [0.0] * (K + 2)
                    cum = 0.0
                    for c in range(K + 1):
                        cum += pk[c]
                        cg[c + 1] = cum
                    Pgd = cg[d] if d <= K else 0.0
                    Egd = sum(n_gc[g][c] for c in range(d, K + 1))
                    gd += (-Egd + Pgd * Ng_g)
                    idd += Ng_g * Pgd * (1.0 - Pgd)
                if idd <= 0:
                    break
                step = gd / idd
                newd = thr[j][m] + step
                newthr = [thr[j][x] if x != m else newd for x in range(K)]
                if _irt_itemQ_poly(a[j], newthr, model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(a[j], thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                    thr[j][m] = newd
                else:
                    imp = False
                    for _h in range(30):
                        step *= 0.5; newd = thr[j][m] + step
                        newthr = [thr[j][x] if x != m else newd for x in range(K)]
                        if _irt_itemQ_poly(a[j], newthr, model_type, Njc[j], nodes, G, K) >= _irt_itemQ_poly(a[j], thr[j], model_type, Njc[j], nodes, G, K) - 1e-12:
                            thr[j][m] = newd; imp = True; break
                    if not imp:
                        break
                if abs(step) < 1e-6:
                    break
            if thr[j][m] < -4.0: thr[j][m] = -4.0
            if thr[j][m] > 4.0: thr[j][m] = 4.0


def _irt_poly_info(aj, thr_j, model_type, theta, K):
    pk = _irt_poly_prob(aj, thr_j, model_type, theta, K)
    if model_type == "pcm":
        Ey = sum(c * pk[c] for c in range(K + 1))
        return sum((c - Ey) ** 2 * pk[c] for c in range(K + 1))
    Gc = [1.0]
    for m in range(1, K + 1):
        Gc.append(_sigmoid(aj * (theta - thr_j[m - 1])))
    I = 0.0
    for c in range(K + 1):
        gc_c = Gc[c]
        gc_c1 = Gc[c + 1] if c + 1 <= K else 0.0
        d = gc_c * (1 - gc_c) - gc_c1 * (1 - gc_c1)
        dlog = aj * d / pk[c] if pk[c] > 1e-12 else 0.0
        I += dlog * dlog * pk[c]
    return I


def _irt_loglik(U, a, b, c, thr, model_type, poly, nodes, wts, G, J, K):
    ll = 0.0
    for i in range(len(U)):
        s = 0.0
        for g in range(G):
            pr = 1.0
            for j in range(J):
                u = U[i][j]
                if u is None:
                    continue
                if poly:
                    pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                    pr *= pk[int(u)]
                else:
                    p = _irt_dich_prob(a[j], b[j], (c[j] if c else 0.0), nodes[g])
                    pr *= (p if u == 1 else (1.0 - p))
            s += wts[g] * pr
        ll += math.log(s + 1e-300)
    return ll


def _irt_reliability(U, a, b, c, thr, model_type, poly, nodes, wts, G, N, J, K):
    vars_ = []
    for i in range(N):
        acc = 0.0; posts = [0.0] * G
        for g in range(G):
            pr = 1.0
            for j in range(J):
                u = U[i][j]
                if u is None:
                    continue
                if poly:
                    pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                    pr *= pk[int(u)]
                else:
                    p = _irt_dich_prob(a[j], b[j], (c[j] if c else 0.0), nodes[g])
                    pr *= (p if u == 1 else (1.0 - p))
            posts[g] = pr * wts[g]; acc += posts[g]
        if acc <= 0:
            vars_.append(1.0); continue
        for g in range(G):
            posts[g] /= acc
        mu = sum(posts[g] * nodes[g] for g in range(G))
        va = sum(posts[g] * (nodes[g] - mu) ** 2 for g in range(G))
        vars_.append(va)
    marg_rel = 1.0 - (sum(vars_) / len(vars_))
    tot_info = [0.0] * G
    for g in range(G):
        I = 0.0
        for j in range(J):
            if poly:
                I += _irt_poly_info(a[j], thr[j], model_type, nodes[g], K)
            else:
                p = _irt_dich_prob(a[j], b[j], (c[j] if c else 0.0), nodes[g])
                I += a[j] ** 2 * p * (1 - p)
        tot_info[g] = I
    marg_info = sum(wts[g] * tot_info[g] for g in range(G))
    info_rel = marg_info / (1.0 + marg_info)
    I0 = 0.0
    for j in range(J):
        if poly:
            I0 += _irt_poly_info(a[j], thr[j], model_type, 0.0, K)
        else:
            p = _irt_dich_prob(a[j], b[j], (c[j] if c else 0.0), 0.0)
            I0 += a[j] ** 2 * p * (1 - p)
    return {"marginal_reliability": round(marg_rel, 4),
            "info_based_reliability": round(info_rel, 4),
            "marginal_test_information": round(marg_info, 4),
            "test_information_at_zero": round(I0, 4)}


def _irt(U, model_type="2pl", n_quad=41, n_iter=100, tol=1e-4, seed=0):
    """项目反应理论（Bock-Aitkin EM 边际极大似然），零依赖。
    支持 1PL(Rasch)/2PL/3PL（0/1 计分）与 GRM/PCM（分类计分 0..K）。
    3PL 含猜测参数 c（∈[0,0.4]）；GRM=等级反应模型，PCM=部分计分模型。
    返回题目参数、项目/测验信息曲线、边际信度（reliability）。"""
    N = len(U)
    if N == 0:
        return {"error": "irt 无有效数据"}
    J = len(U[0])
    if J < 2:
        return {"error": "irt 至少需要 2 个题目"}
    poly = model_type in ("grm", "pcm")
    K = 0
    if poly:
        K = 0
        for row in U:
            for v in row:
                if v is None:
                    continue
                if v != int(v):
                    return {"error": "polytomous IRT 要求整数计分(0..K)"}
                iv = int(v)
                if iv < 0:
                    return {"error": "polytomous IRT 计分须为非负整数"}
                if iv > K:
                    K = iv
        if K < 1:
            return {"error": "polytomous IRT 需 ≥2 个类别（计分 0..K，K≥1）"}
    else:
        for row in U:
            for v in row:
                if v is None:
                    continue
                if v not in (0, 1):
                    return {"error": "1PL/2PL/3PL 要求 0/1 计分"}
    nodes, wts = _gauss_hermite(n_quad)
    G = len(nodes)
    rng = random.Random(seed)
    is_3pl = (model_type == "3pl")
    a = [1.0 for _ in range(J)]
    b = [0.0 for _ in range(J)]
    c = [0.2 for _ in range(J)] if is_3pl else None
    thr = None
    if poly:
        thr = [[(m - (K + 1) / 2.0) * 0.6 for m in range(1, K + 1)] for _ in range(J)]
    prev_ll = None
    for it in range(n_iter):
        r = [[0.0] * G for _ in range(N)]
        if poly:
            Njc = [[dict() for _ in range(G)] for _ in range(J)]
        else:
            Ug = [[0.0] * G for _ in range(J)]
        Ng = [0.0] * G
        for i in range(N):
            acc = 0.0
            for g in range(G):
                pr = 1.0
                for j in range(J):
                    u = U[i][j]
                    if u is None:
                        continue
                    if poly:
                        pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                        pr *= pk[int(u)]
                    else:
                        p = _irt_dich_prob(a[j], b[j], (c[j] if is_3pl else 0.0), nodes[g])
                        pr *= (p if u == 1 else (1.0 - p))
                r[i][g] = pr * wts[g]
                acc += r[i][g]
            if acc <= 0:
                continue
            for g in range(G):
                r[i][g] /= acc
                w = r[i][g]
                Ng[g] += w
                for j in range(J):
                    u = U[i][j]
                    if u is None:
                        continue
                    if poly:
                        d = Njc[j][g]
                        d[int(u)] = d.get(int(u), 0.0) + w
                    else:
                        Ug[j][g] += w * u
        if poly:
            for j in range(J):
                _irt_poly_mstep(j, thr, a, model_type, Njc, Ng, nodes, G, K)
        else:
            for j in range(J):
                _irt_dich_mstep(j, a, b, c, is_3pl, Ug, Ng, nodes, G, model_type)
        ll = _irt_loglik(U, a, b, c, thr, model_type, poly, nodes, wts, G, J, K)
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll
    items = []
    for j in range(J):
        if poly:
            curve = []
            for g in range(G):
                pk = _irt_poly_prob(a[j], thr[j], model_type, nodes[g], K)
                info = _irt_poly_info(a[j], thr[j], model_type, nodes[g], K)
                curve.append({"theta": round(nodes[g], 3), "probs": [round(x, 4) for x in pk], "info": round(info, 4)})
            items.append({"item": j, "a": round(a[j], 4),
                          "thresholds": [round(x, 4) for x in thr[j]],
                          "max_info": round(max((cc["info"] for cc in curve), default=0.0), 4),
                          "curve": curve})
        else:
            curve = []
            for g in range(G):
                P = _irt_dich_prob(a[j], b[j], (c[j] if is_3pl else 0.0), nodes[g])
                I = a[j] ** 2 * P * (1 - P)
                curve.append({"theta": round(nodes[g], 3), "prob": round(P, 4), "info": round(I, 4)})
            it = {"item": j, "a": round(a[j], 4), "b": round(b[j], 4),
                  "max_info": round(a[j] ** 2 * 0.25, 4), "curve": curve}
            if is_3pl:
                it["c"] = round(c[j], 4)
            items.append(it)
    total_info = []
    for g in range(G):
        I = 0.0
        for j in range(J):
            if poly:
                I += _irt_poly_info(a[j], thr[j], model_type, nodes[g], K)
            else:
                P = _irt_dich_prob(a[j], b[j], (c[j] if is_3pl else 0.0), nodes[g])
                I += a[j] ** 2 * P * (1 - P)
        total_info.append(round(I, 4))
    rel = _irt_reliability(U, a, b, c, thr, model_type, poly, nodes, wts, G, N, J, K)
    theta_grid = [round(nodes[g], 3) for g in range(G)]
    sep = {"1pl": "1PL(Rasch)", "2pl": "2PL", "3pl": "3PL", "grm": "GRM(等级反应)", "pcm": "PCM(部分计分)"}[model_type]
    if poly:
        interp = ("项目反应理论（%s，EM 估计）：%d 题 / %d 人 / %d 类别；边际信度=%.3f，测验信息@0=%.2f。"
                  % (sep, J, N, K + 1, rel["marginal_reliability"], rel["test_information_at_zero"]))
    else:
        bs = sorted(round(b[j], 3) for j in range(J))
        as_ = [round(a[j], 3) for j in range(J)]
        interp = ("项目反应理论（%s，EM 估计）：%d 题 / %d 人；难度 b 范围 [%.2f,%.2f]，平均区分度 a=%.2f；"
                  "边际信度=%.3f。" % (sep, J, N, min(bs), max(bs),
                                      (sum(as_) / len(as_) if as_ else 0.0), rel["marginal_reliability"]))
    res = {"type": "irt", "model": sep, "n_persons": N, "n_items": J,
           "items": items, "theta_grid": theta_grid, "total_information": total_info,
           "reliability": rel, "interpretation": interp}
    if poly:
        res["n_categories"] = K + 1
    return res

def cmd_irt(args):
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    item_cols = [c.strip() for c in args.items.split(",") if c.strip()]
    if len(item_cols) < 2:
        die("irt 需至少 2 个题目列（--items a,b,c,...）")
    data = []
    poly = args.model_type in ("grm", "pcm")
    for r in rows:
        row = [to_float(r.get(c)) for c in item_cols]
        if any(x is None for x in row):
            continue
        if poly:
            # 分类计分（0/1/2/...）保留整数，供 GRM/PCM 使用
            data.append([int(round(x)) for x in row])
        else:
            data.append([1 if x >= 0.5 else 0 for x in row])
    if len(data) < 10:
        die("irt 有效样本不足（有效行 < 10）")
    res = _irt(data, args.model_type)
    if "error" in res:
        die(res["error"])
    emit({"status": "ok", "task": "irt", "result": res})


# ----- T5 因子分析(EFA) + McDonald's ω 组合信度 + 线性混合模型(LMM) -----
def _eigh_sym(A):
    """对称矩阵特征分解（Jacobi 旋转，纯 Python；numpy 可用时优先）。返回 (eigenvalues 降序, eigenvectors[列][行])。"""
    if HAS_NUMPY:
        try:
            w, V = np.linalg.eigh(np.array(A, dtype=float))
            order = sorted(range(len(w)), key=lambda k: -w[k])
            evals = [float(w[k]) for k in order]
            evecs = [[float(V[i][order[j]]) for i in range(len(V))] for j in range(len(V))]
            return evals, evecs
        except Exception:
            pass
    n = len(A)
    a = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(200):
        p = q = 0; maxv = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > maxv:
                    maxv = abs(a[i][j]); p = i; q = j
        if maxv < 1e-12:
            break
        phi = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c = math.cos(phi); s = math.sin(phi)
        for i in range(n):
            ai_p, ai_q = a[i][p], a[i][q]
            a[i][p] = c * ai_p - s * ai_q
            a[i][q] = s * ai_p + c * ai_q
        for i in range(n):
            ai_p, ai_q = a[p][i], a[q][i]
            a[p][i] = c * ai_p - s * ai_q
            a[q][i] = s * ai_p + c * ai_q
        for i in range(n):
            vi_p, vi_q = V[i][p], V[i][q]
            V[i][p] = c * vi_p - s * vi_q
            V[i][q] = s * vi_p + c * vi_q
    evals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda k: -evals[k])
    evals = [evals[k] for k in order]
    evecs = [[V[i][order[j]] for i in range(n)] for j in range(n)]
    return evals, evecs


def _varimax(L, gamma=1.0, max_iter=200, tol=1e-8):
    """Kaiser 方差最大旋转（基于极分解/SVD，对对称结构稳健）。L: items×factors 载荷矩阵，返回旋转后载荷。"""
    p = len(L); k = len(L[0])
    R = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for _ in range(max_iter):
        Lam = [[sum(L[r][a] * R[a][c] for a in range(k)) for c in range(k)] for r in range(p)]
        colss = [sum(Lam[r][c] ** 2 for r in range(p)) for c in range(k)]
        B = [[Lam[r][c] ** 3 - (gamma / p) * Lam[r][c] * colss[c] for c in range(k)] for r in range(p)]
        M = [[sum(L[r][a] * B[r][c] for r in range(p)) for c in range(k)] for a in range(k)]
        Mt = [[M[c][a] for c in range(k)] for a in range(k)]
        MtM = [[sum(Mt[a][b] * M[b][c] for b in range(k)) for c in range(k)] for a in range(k)]
        evals, evecs = _eigh_sym(MtM)
        sqinv = [[0.0] * k for _ in range(k)]
        for a in range(k):
            inv = 1.0 / math.sqrt(max(evals[a], 1e-12))
            for i in range(k):
                for j in range(k):
                    sqinv[i][j] += evecs[i][a] * inv * evecs[j][a]
        Rnew = [[sum(M[a][b] * sqinv[b][c] for b in range(k)) for c in range(k)] for a in range(k)]
        diff = sum(abs(Rnew[a][c] - R[a][c]) for a in range(k) for c in range(k))
        R = Rnew
        if diff < tol:
            break
    return [[sum(L[r][a] * R[a][c] for a in range(k)) for c in range(k)] for r in range(p)]


def _pairwise_corr(feats, rows):
    """特征两两 Pearson 相关矩阵（成对完整）。返回 (R, degenerate)；degenerate 为方差=0/样本不足的特征名。"""
    series = {f: [to_float(r.get(f)) for r in rows] for f in feats}
    p = len(feats)
    R = [[0.0] * p for _ in range(p)]
    degenerate = []
    for a in range(p):
        xs = [x for x in series[feats[a]] if x is not None]
        if len(xs) < 3 or stdev(xs) <= 0:
            degenerate.append(feats[a])
    for a in range(p):
        for b in range(a, p):
            if feats[a] in degenerate or feats[b] in degenerate:
                r = 0.0  # 含零方差/样本不足特征时相关性无定义，置 0
            else:
                xa = series[feats[a]]; xb = series[feats[b]]
                pairs = [(xa[i], xb[i]) for i in range(len(xa)) if xa[i] is not None and xb[i] is not None]
                r = pearson([t[0] for t in pairs], [t[1] for t in pairs]) if len(pairs) >= 3 else 0.0
            R[a][b] = R[b][a] = (r if r is not None else 0.0)
    return R, degenerate


def _mat_logdet(A):
    """返回 (log|det|, sign)。零依赖 LU 偏主元。PD 矩阵 sign=+1；奇异返回 None。"""
    n = len(A)
    M = [row[:] for row in A]
    scale = max((max(abs(v) for v in row) for row in M), default=0.0)
    if scale == 0.0:
        return None
    thresh = scale * 1e-12
    sign = 1.0
    logd = 0.0
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        if piv != i:
            M[i], M[piv] = M[piv], M[i]
            sign = -sign
        d = M[i][i]
        if abs(d) < thresh:
            return None
        logd += math.log(abs(d))
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n):
                M[r][c] -= f * M[i][c]
    return logd, sign


def _nelder_mead(f, x0, max_iter=600, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """零依赖 Nelder-Mead 下山单纯形优化（小参数量问题稳健）。f 应返回标量（越大越差）。"""
    BIG = 1e18
    def _f(x):
        try:
            v = f(x)
        except Exception:
            return BIG
        if v is None:
            return BIG
        return v
    n = len(x0)
    simplex = [list(x0)]
    fvals = [_f(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += 0.1
        simplex.append(x)
        fvals.append(_f(x))
    for _ in range(max_iter):
        order = sorted(range(len(simplex)), key=lambda k: fvals[k])
        if abs(fvals[order[-1]] - fvals[order[0]]) < tol and _ > 20:
            spread = max(max(abs(simplex[order[-1]][i] - simplex[order[0]][i]) for i in range(n)) for _d in range(1))
            if spread < 1e-4:
                break
        centroid = [sum(simplex[order[k]][i] for k in range(n)) / n for i in range(n)]
        xr = [centroid[i] + alpha * (centroid[i] - simplex[order[-1]][i]) for i in range(n)]
        fr = _f(xr)
        if fr < fvals[order[0]]:
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(n)]
            fe = _f(xe)
            if fe < fr:
                simplex[order[-1]], fvals[order[-1]] = xe, fe
            else:
                simplex[order[-1]], fvals[order[-1]] = xr, fr
        elif fr < fvals[order[-2]]:
            simplex[order[-1]], fvals[order[-1]] = xr, fr
        else:
            if fr < fvals[order[-1]]:
                xc = [centroid[i] + rho * (xr[i] - centroid[i]) for i in range(n)]
                fc = _f(xc)
                if fc < fr:
                    simplex[order[-1]], fvals[order[-1]] = xc, fc
                else:
                    for k in range(1, len(simplex)):
                        idx = order[k]
                        simplex[idx] = [simplex[order[0]][i] + sigma * (simplex[idx][i] - simplex[order[0]][i]) for i in range(n)]
                        fvals[idx] = _f(simplex[idx])
            else:
                xc = [centroid[i] + rho * (simplex[order[-1]][i] - centroid[i]) for i in range(n)]
                fc = _f(xc)
                if fc < fvals[order[-1]]:
                    simplex[order[-1]], fvals[order[-1]] = xc, fc
                else:
                    for k in range(1, len(simplex)):
                        idx = order[k]
                        simplex[idx] = [simplex[order[0]][i] + sigma * (simplex[idx][i] - simplex[order[0]][i]) for i in range(n)]
                        fvals[idx] = _f(simplex[idx])
    order = sorted(range(len(simplex)), key=lambda k: fvals[k])
    return simplex[order[0]], fvals[order[0]]


def _parse_cfa_model(spec, feats):
    """解析 CFA 模型规格 'F1:i1,i2,i3;F2:i4,i5,i6' → [(fname,[item索引]...), ...]。"""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        fname, items = part.split(":", 1)
        idx = []
        for it in items.split(","):
            it = it.strip()
            if it in feats:
                idx.append(feats.index(it))
            else:
                return None, "模型含未知变量: %s" % it
        if idx:
            out.append((fname.strip(), idx))
    if not out:
        return None, "模型为空或解析失败"
    return out, None


def _cfa(R, n_obs, factors, orthogonal, feats):
    """验证性因子分析（CFA，相关矩阵尺度，零依赖）。
    自由载荷 λ 经 tanh 参数化 → |λ|<1（自动满足载荷≤1，Heywood 病可检出）；
    唯一方差 ψ_i 由单位对角约束导出：ψ_i = 1 − (Σ_q λ² + 2Σ_{q<r} λ_q λ_r φ_qr)。
    最小化 ML 差异函数 F = log|Σ| + tr(R Σ⁻¹) − log|R| − p（Σ = ΛΦΛᵀ + Ψ）。
    返回载荷(标准化)、因子相关、因子方差(1)、拟合指数 CFI/TLI/RMSEA/SRMR、Heywood 标记。"""
    p = len(R)
    q = len(factors)
    load_map = [[0.0] * q for _ in range(p)]
    for f, (_, items) in enumerate(factors):
        for i in items:
            load_map[i][f] = 1.0
    free_load = [(i, f) for i in range(p) for f in range(q) if load_map[i][f] == 1.0]
    n_load = len(free_load)
    phi_pairs = []
    if not orthogonal and q > 1:
        phi_pairs = [(f1, f2) for f1 in range(q) for f2 in range(f1 + 1, q)]
    n_params = n_load + len(phi_pairs)

    def unpack(x):
        lam = [[0.0] * q for _ in range(p)]
        for k, (i, f) in enumerate(free_load):
            lam[i][f] = math.tanh(x[k])
        phi = [[0.0] * q for _ in range(q)]
        for f in range(q):
            phi[f][f] = 1.0
        for kk, (f1, f2) in enumerate(phi_pairs):
            v = math.tanh(x[n_load + kk])
            phi[f1][f2] = v
            phi[f2][f1] = v
        return lam, phi

    def build_sigma(lam, phi):
        Lam = [[lam[i][f] for f in range(q)] for i in range(p)]
        com = [0.0] * p
        for i in range(p):
            s = 0.0
            for f in range(q):
                s += lam[i][f] ** 2
            for f1 in range(q):
                for f2 in range(f1 + 1, q):
                    s += 2.0 * lam[i][f1] * lam[i][f2] * phi[f1][f2]
            com[i] = s
        psi = [max(1.0 - com[i], 1e-3) for i in range(p)]
        Sig = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                s = 0.0
                for f1 in range(q):
                    for f2 in range(q):
                        s += Lam[i][f1] * phi[f1][f2] * Lam[j][f2]
                Sig[i][j] = s
        for i in range(p):
            Sig[i][i] += psi[i]
        return Sig, psi

    ldR = _mat_logdet(R)
    if ldR is None:
        return {"error": "相关矩阵奇异，CFA 无法估计"}
    logdetR = ldR[0]
    F0 = -logdetR  # 独立性(零)模型差异 = -log|R|
    df_0 = p * (p - 1) // 2

    def objective(x):
        lam, phi = unpack(x)
        Sig, _ = build_sigma(lam, phi)
        ld = _mat_logdet(Sig)
        if ld is None:
            return 1e9
        Sinv = _mat_inv(Sig)
        if Sinv is None:
            return 1e9
        tr = 0.0
        for i in range(p):
            for j in range(p):
                tr += R[i][j] * Sinv[j][i]
        return ld[0] + tr - logdetR - p

    x0 = [0.3] * n_params  # 温和初始载荷
    xopt = _nelder_mead(objective, x0)[0]
    lam, phi = unpack(xopt)
    Sig, psi = build_sigma(lam, phi)
    Fmin = objective(xopt)
    T = n_obs * Fmin
    df_model = p * (p + 1) // 2 - n_params
    T0 = n_obs * F0
    cfi = 1.0 - max(T - df_model, 0.0) / max(T0 - df_0, 0.0) if (T0 - df_0) > 0 else None
    cfi = min(1.0, cfi) if cfi is not None else None
    tli = (T0 / df_0 - T / df_model) / (T0 / df_0 - 1.0) if df_model > 0 and (T0 / df_0 - 1.0) != 0 else None
    tli = min(1.0, tli) if tli is not None else None
    rmsea = math.sqrt(max(T / df_model - 1.0, 0.0) / (n_obs - 1)) if df_model > 0 else None
    ssum = 0.0
    for i in range(p):
        for j in range(i + 1):
            d = R[i][j] - Sig[i][j]
            ssum += d * d
    srmr = math.sqrt(2.0 * ssum / (p * (p + 1)))
    # 载荷超限(Heywood)检测
    heywood = []
    for k, (i, f) in enumerate(free_load):
        if abs(math.tanh(xopt[k])) >= 0.98 or psi[i] < 0.02:
            heywood.append(feats[i])
    loadings_out = []
    for f, (fname, items) in enumerate(factors):
        for i in items:
            loadings_out.append({"factor": fname, "item": feats[i],
                                  "loading": round(lam[i][f], 4),
                                  "std_loading": round(lam[i][f], 4)})
    fac_cov = None
    if not orthogonal and q > 1:
        fac_cov = {"factors": [fname for fname, _ in factors],
                   "matrix": [[round(phi[f1][f2], 4) for f2 in range(q)] for f1 in range(q)]}
    return {
        "n_items": p, "n_factors": q, "n_obs": n_obs, "orthogonal": orthogonal,
        "loadings": loadings_out,
        "factor_covariance": fac_cov,
        "uniqueness": {feats[i]: round(psi[i], 4) for i in range(p)},
        "fit": {
            "discrepancy": round(Fmin, 6),
            "chi_square": round(T, 4), "df": df_model,
            "CFI": round(cfi, 4) if cfi is not None else None,
            "TLI": round(tli, 4) if tli is not None else None,
            "RMSEA": round(rmsea, 4) if rmsea is not None else None,
            "SRMR": round(srmr, 4),
        },
        "heywood_cases": heywood if heywood else None,
        "note": ("CFA（相关矩阵尺度，ML 估计）：载荷经 tanh 参数化自动约束 |λ|<1；"
                 "CFI≥0.95/TLI≥0.95/RMSEA≤0.06/SRMR≤0.08 通常判为可接受拟合；"
                 "Heywood 病（载荷≥0.98 或唯一方差<0.02）提示模型设定或样本问题。"),
    }


def cmd_factor(args):
    """因子分析：EFA（主成分/主轴 + varimax）、McDonald's ω 组合信度、CFA（验证性因子分析，载荷≤1 + CFI/TLI/RMSEA/SRMR）。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    if len(feats) < 3:
        die("因子分析需 ≥3 个变量（--features）")
    for f in feats:
        if f not in columns:
            die("变量不存在: %s" % f)
    R, degenerate = _pairwise_corr(feats, rows)
    if getattr(args, "mode", "efa") == "cfa":
        spec = getattr(args, "cfa_model", None)
        if not spec:
            die("CFA 模式需 --cfa-model 指定模型，如 'F1:i1,i2,i3;F2:i4,i5,i6'")
        factors, err = _parse_cfa_model(spec, feats)
        if err:
            die(err)
        n_obs = sum(1 for r in rows if all(to_float(r.get(f)) is not None for f in feats))
        if n_obs < len(feats) + 1:
            die("CFA 有效观测不足（需 > 变量数）")
        orthogonal = getattr(args, "cfa_orthogonal", False)
        res = _cfa(R, n_obs, factors, orthogonal, feats)
        if "error" in res:
            die(res["error"])
        fit = res["fit"]
        verdict = ("拟合优良" if (fit.get("CFI") or 0) >= 0.95 and (fit.get("RMSEA") or 1) <= 0.06
                   else "拟合尚可" if (fit.get("CFI") or 0) >= 0.90 else "拟合不佳")
        res["interpretation"] = ("验证性因子分析（%s，%d 因子，n=%d）：CFI=%.3f，TLI=%.3f，RMSEA=%.3f，SRMR=%.3f；%s。"
                                 % (("正交" if orthogonal else "斜交"), res["n_factors"], n_obs,
                                    fit.get("CFI") or 0, fit.get("TLI") or 0,
                                    fit.get("RMSEA") or 0, fit.get("SRMR") or 0, verdict))
        if res.get("heywood_cases"):
            _warn("CFA 检出 Heywood 病（载荷≥0.98 或唯一方差<0.02）：%s" % "、".join(res["heywood_cases"]))
        emit({"status": "ok", "task": "factor", "result": res}); return
    n = len(R)
    evals, evecs = _eigh_sym(R)
    if getattr(args, "mode", "efa") == "reliability":
        m = 1  # 单因子解估计组合信度
    elif args.n_factors and args.n_factors > 0:
        m = min(args.n_factors, n)
    else:
        m = max(1, sum(1 for e in evals if e > 1.0))
    m = min(m, n)
    load = [[evecs[j][i] * math.sqrt(max(evals[j], 0.0)) for j in range(m)] for i in range(n)]
    total_var = sum(evals) if sum(evals) > 0 else 1.0
    explained = [round(evals[j] / total_var, 4) for j in range(m)]
    rotated = _varimax(load) if (args.rotation == "varimax" and m > 1) else load
    commun = [round(sum(rotated[i][a] ** 2 for a in range(m)), 4) for i in range(n)]
    uniq = [round(1.0 - commun[i], 4) for i in range(n)]
    out = {"mode": args.mode, "n_factors": m, "features": feats,
           "degenerate_features": degenerate,
           "eigenvalues": [round(e, 4) for e in evals],
           "variance_explained": explained,
           "loadings": [dict(zip(["f%d" % (a + 1) for a in range(m)],
                                [round(rotated[i][a], 4) for a in range(m)])) for i in range(n)],
           "communalities": dict(zip(feats, commun)),
           "uniqueness": dict(zip(feats, uniq)),
           "rotation": "varimax" if m > 1 else "none"}
    if args.mode == "reliability":
        # 单通用因子解估计组合信度：取首主成分载荷（取绝对值以保证载荷同向，符合"通用因子"语义）
        lam = [abs(load[i][0]) for i in range(n)]
        psi = [1.0 - lam[i] ** 2 for i in range(n)]
        num = sum(lam) ** 2
        den = num + sum(psi)
        omega = round(num / den, 4) if den > 0 else None
        out["mcdonalds_omega"] = omega
        out["note"] = "reliability 模式：以单因子解为通用因子估计 McDonald's ω（组合信度，通常优于 Cronbach's α）"
        out["interpretation"] = "McDonald's ω = %.3f（>0.7 表示组合信度可接受）" % (omega if omega is not None else 0.0)
    else:
        out["note"] = "EFA（主成分/主轴 + 可选 varimax）；载荷为旋转后载荷，communalities/uniqueness 为公因子方差与唯一性"
        # 生成可读结论：每因子的主要载荷题项 + 累计解释方差（供 narrate 复用，与全内核一致）
        cum = round(sum(explained), 4)
        factor_terms = []
        for a in range(m):
            loaded = [feats[i] for i in range(n) if abs(rotated[i][a]) >= 0.4]
            if not loaded:
                loaded = [feats[i] for i in sorted(range(n), key=lambda i: -abs(rotated[i][a]))[:3]]
            factor_terms.append("因子%d：%s" % (a + 1, "、".join(loaded)))
        out["interpretation"] = (
            "探索性因子分析（%s 旋转）提取 %d 个因子，累计解释方差 %.1f%%；%s。"
            % (out["rotation"], m, cum * 100.0, "；".join(factor_terms)))
    emit({"status": "ok", "task": "factor", "result": out})


def _lmm_group(Xg, yg, Zg, s2, sb2, ss2, xb=None):
    """单组 V_g⁻¹ 相关量：返回 (XtVX, XtVy, Vinv_y, Minv, ZtVinvZ, C, trCZtZ)；奇异返回 None。
    xb: 该组固定效应拟合值（长度 ng）；若提供，则 Vinv_y 返回 V^{-1}(y-Xβ)（BLUP 居中对）。
    C = Var(b|y) = G - G Z'V^{-1}Z G（真·条件协方差；验证：单随机截距 n_g 组
    posterior = (1/sb2 + n_g/s2)^{-1}，等于 sb2 - sb2^2·Z'V^{-1}Z[0][0]）；
    trCZtZ = tr(C·Z'Z)，为 EM 残差更新的校正项 tr(Z C Z')。"""
    ng = len(yg); p = len(Xg[0]); r = len(Zg[0])
    Zt = [[Zg[j][a] for j in range(ng)] for a in range(r)]
    ZtZ = [[sum(Zt[a][i] * Zg[i][b] for i in range(ng)) for b in range(r)] for a in range(r)]
    Gdiag = [sb2] + ([ss2] if r > 1 else [])
    Ginv = [[(1.0 / Gdiag[a] if a == b else 0.0) for b in range(r)] for a in range(r)]
    M2 = [[ZtZ[a][b] + s2 * Ginv[a][b] for b in range(r)] for a in range(r)]
    Minv = _mat_inv(M2)
    if Minv is None:
        return None
    ZM = [[sum(Zg[i][a] * Minv[a][b] for a in range(r)) for b in range(r)] for i in range(ng)]
    ZMZt = [[sum(ZM[i][a] * Zg[j][a] for a in range(r)) for j in range(ng)] for i in range(ng)]
    Vinv = [[((1.0 if i == j else 0.0) - ZMZt[i][j]) / s2 for j in range(ng)] for i in range(ng)]
    # Z'V^{-1}Z（r×r），用于 EM 方差组分更新
    ZtVinvZ = [[sum(Zg[i][a] * Vinv[i][j] * Zg[j][c] for i in range(ng) for j in range(ng))
                for c in range(r)] for a in range(r)]
    # 条件协方差 C = G - G Z'V^{-1}Z G（G 对角）
    C = [[(Gdiag[a] if a == b else 0.0) - Gdiag[a] * ZtVinvZ[a][b] * Gdiag[b]
          for b in range(r)] for a in range(r)]
    trCZtZ = sum(C[a][b] * ZtZ[b][a] for a in range(r) for b in range(r))
    Xt = [[Xg[j][a] for j in range(ng)] for a in range(p)]
    XtVX = [[sum(Xt[a][i] * sum(Vinv[i][k] * Xg[k][b] for k in range(ng)) for i in range(ng)) for b in range(p)]
            for a in range(p)]
    XtVy = [sum(Xt[a][i] * sum(Vinv[i][k] * yg[k] for k in range(ng)) for i in range(ng)) for a in range(p)]
    if xb is not None:
        Vinv_y = [sum(Vinv[i][k] * (yg[k] - xb[k]) for k in range(ng)) for i in range(ng)]
    else:
        Vinv_y = [sum(Vinv[i][k] * yg[k] for k in range(ng)) for i in range(ng)]
    return XtVX, XtVy, Vinv_y, Minv, ZtVinvZ, C, trCZtZ


def _lmm(y, X, groups, names, slope=None, max_iter=200, tol=1e-8):
    """随机截距(±随机斜率)线性混合模型：MME + EM 估计方差组分。返回结果 dict。"""
    n = len(y); p = len(X[0]); gvals = sorted(set(groups)); g = len(gvals)
    if n < p + 2 + g:
        return {"error": "样本量不足以拟合 LMM（需 > %d）" % (p + g + 1)}
    idx_by_g = {gv: [i for i in range(n) if groups[i] == gv] for gv in gvals}
    v = stdev(y) if stdev(y) > 0 else 1.0
    sb2 = 0.5 * v ** 2; s2 = v ** 2; ss2 = 0.1 * s2
    has_slope = slope is not None
    beta = [0.0] * p
    for _ in range(max_iter):
        # 第一遍：累积 X'V^{-1}X、X'V^{-1}y 估计固定效应 β（GLS）
        XtVX = [[0.0] * p for _ in range(p)]
        XtVy = [0.0] * p
        sb2 = max(sb2, 1e-8); s2 = max(s2, 1e-8); ss2 = max(ss2, 1e-8)
        for gv in gvals:
            gi = idx_by_g[gv]
            Xg = [X[i] for i in gi]; yg = [y[i] for i in gi]
            Zg = [[1.0, slope[i]] for i in gi] if has_slope else [[1.0] for i in gi]
            r = _lmm_group(Xg, yg, Zg, s2, sb2, ss2)
            if r is None:
                return {"error": "LMM 矩阵奇异"}
            aXtVX, aXtVy, *_ = r
            for a in range(p):
                XtVy[a] += aXtVy[a]
                for b in range(p):
                    XtVX[a][b] += aXtVX[a][b]
        inv = _mat_inv(XtVX)
        if inv is None:
            return {"error": "X'V⁻¹X 奇异"}
        beta = [sum(inv[a][b] * XtVy[b] for b in range(p)) for a in range(p)]
        # 第二遍：用已居中的 V^{-1}(y-Xβ) 计算 BLUP 与方差组分（EM）
        sb2_n = ss2_n = s2_num = 0.0
        for gv in gvals:
            gi = idx_by_g[gv]
            Xg = [X[i] for i in gi]; yg = [y[i] for i in gi]
            Zg = [[1.0, slope[i]] for i in gi] if has_slope else [[1.0] for i in gi]
            xb = [sum(Xg[i][a] * beta[a] for a in range(p)) for i in range(len(gi))]
            r = _lmm_group(Xg, yg, Zg, s2, sb2, ss2, xb=xb)
            if r is None:
                return {"error": "LMM 矩阵奇异"}
            aXtVX, aXtVy, _, Minv, _, C, trCZtZ = r
            # BLUP: b̂ = Minv·Zᵀ(y−Xβ)。Minv 已含 V⁻¹（恒等式
            # (ZᵀZ+σ²G⁻¹)⁻¹Zᵀ = GZᵀV⁻¹），故 Zty 用原始居中和 Zᵀ(y−xb)，不可再乘 V⁻¹。
            Zty = [sum(Zg[i][a] * (yg[i] - xb[i]) for i in range(len(gi))) for a in range(len(Zg[0]))]
            bg = [sum(Minv[a][b] * Zty[b] for b in range(len(Zg[0]))) for a in range(len(Zg[0]))]
            # EM 方差组分（M-step）：E[u²|y] = b̂² + Var(u|y)；C 为真条件协方差 G−GZ'V⁻¹ZG
            sb2_n += bg[0] ** 2 + C[0][0]
            if has_slope:
                ss2_n += bg[1] ** 2 + C[1][1]
            # EM 残差校正：E[‖y−Xβ−Zu‖²|y] = ‖(y−Xβ)−Zb̂‖² + tr(Z C Z') = 残差平方和 + tr(C·Z'Z)
            s2_num += trCZtZ
            for i in range(len(gi)):
                pred = sum(Xg[i][a] * beta[a] for a in range(p)) + bg[0] + (slope[i] * bg[1] if has_slope else 0.0)
                s2_num += (yg[i] - pred) ** 2
        sb2_new = sb2_n / g
        ss2_new = ss2_n / g if has_slope else ss2
        s2_new = s2_num / n if n > 0 else s2_num
        if (abs(sb2_new - sb2) / max(sb2, 1e-9) < tol and
                abs(s2_new - s2) / max(s2, 1e-9) < tol and
                abs(ss2_new - ss2) / max(ss2, 1e-9) < tol):
            sb2, s2, ss2 = sb2_new, s2_new, ss2_new
            break
        sb2, s2, ss2 = sb2_new, s2_new, ss2_new
    inv = _mat_inv(XtVX)
    se = [math.sqrt(inv[a][a]) if inv and inv[a][a] > 0 else None for a in range(p)]
    coefs = []
    for a in range(p):
        z = beta[a] / se[a] if se[a] else None
        pval = _norm_sf(abs(z)) * 2.0 if z is not None else None
        coefs.append({"term": names[a], "coef": round(beta[a], 4),
                      "se": round(se[a], 4) if se[a] else None,
                      "z": round(z, 3) if z is not None else None,
                      "p_value": _r4(pval) if pval is not None else None,
                      "significant": _sig(pval)})
    icc = round(sb2 / (sb2 + s2), 4) if (sb2 + s2) > 0 else None
    vc = {"residual": round(s2, 4), "random_intercept": round(sb2, 4)}
    if has_slope:
        vc["random_slope"] = round(ss2, 4)
    return {"fixed_effects": coefs, "variance_components": vc, "icc": icc,
            "n_groups": g, "n": n,
            "note": "随机截距(±斜率)线性混合模型，MME+EM 估计；固定效应 Wald z 检验（大样本正态近似）",
            "interpretation": "ICC=%.3f（组间方差占比）；固定效应见 fixed_effects。" % (icc if icc is not None else 0.0)}


def cmd_lmm(args):
    """线性混合模型：固定效应 + 随机截距(±随机斜率)，适用于重复测量/纵向/分层数据。"""
    _apply_cfg(args)
    columns, rows = load_rows(args.input)
    ycol = args.y
    if ycol not in columns:
        die("因变量字段不存在: %s" % ycol)
    xcols = [c.strip() for c in args.x.split(",") if c.strip()]
    if not xcols:
        die("需至少 1 个固定效应（--x）")
    for c in xcols:
        if c not in columns:
            die("自变量不存在: %s" % c)
    gcol = args.group
    if gcol not in columns:
        die("分组列(随机截距)不存在: %s" % gcol)
    scol = getattr(args, "random_slope", None)
    if scol and scol not in columns:
        die("随机斜率列不存在: %s" % scol)
    y = []; X = []; groups = []; slope = []
    for r in rows:
        yv = to_float(r.get(ycol))
        xv = [to_float(r.get(c)) for c in xcols]
        gv = r.get(gcol)
        if yv is None or any(v is None for v in xv) or gv is None or str(gv).strip() == "":
            continue
        y.append(yv); X.append([1.0] + xv); groups.append(str(gv))
        if scol:
            sv = to_float(r.get(scol))
            slope.append(sv if sv is not None else 0.0)
    names = ["intercept"] + xcols
    res = _lmm(y, X, groups, names, slope if scol else None)
    if "error" in res:
        emit({"status": "error", "task": "lmm", "message": res["error"]}); return
    emit({"status": "ok", "task": "lmm", "result": res})


# ----------------------------------------------------------------------------
# Batch 3：runs（跨运行溯源报告对比 / diff） + narrate（APA 式结果草稿）
# 设计原则：复用既有 report/stats 输出的 JSON 字段与 interpretation 文本，
# 不重新计算统计量，避免冗余逻辑。
# ----------------------------------------------------------------------------

def _load_unwrap(path):
    """读取 JSON，解开 {status,task,result} 外层，返回 (task, 内层 dict)。"""
    if not os.path.exists(path):
        return None, {"__error__": "文件不存在: %s" % path}
    try:
        size = os.path.getsize(path)
        if size > _MAX_INPUT_BYTES:
            return None, {"__error__": "文件过大（%d 字节 > 上限 %d 字节），已拒绝" % (size, _MAX_INPUT_BYTES)}
    except OSError:
        pass
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
    except Exception as e:
        return None, {"__error__": "无法读取 %s: %s" % (path, e)}
    if not isinstance(obj, dict):
        return None, {"__error__": "非 JSON 对象: %s" % path}
    task = obj.get("task")
    inner = obj
    if "result" in obj and isinstance(obj["result"], dict) and \
            set(obj.keys()) <= {"status", "task", "result"}:
        inner = obj["result"]
    if task is None and isinstance(inner, dict):
        task = inner.get("task")
    return task, inner


def _report_fields(rep):
    """从单个 report 结果 dict 抽取跨运行对比所需字段。"""
    env = rep.get("environment") or {}
    ds = rep.get("dataset") or {}
    cols = ds.get("columns") or []
    num = ds.get("numeric_columns") or []
    anom = rep.get("anomaly_iqr_L2") or {}
    total_anom = 0
    for a in anom.values():
        total_anom += (a.get("count") if isinstance(a, dict) else (a or 0))
    return {
        "run_id": rep.get("run_id"),
        "environment": {
            "python_version": env.get("python_version"),
            "numpy": env.get("numpy"),
            "scipy": env.get("scipy"),
            "sklearn": env.get("sklearn"),
            "zero_dependency_core": env.get("zero_dependency_core"),
        },
        "rows": ds.get("rows"),
        "n_columns": len(cols) if isinstance(cols, list) else None,
        "n_numeric": len(num) if isinstance(num, list) else None,
        "n_anomaly": total_anom,
        "summary": rep.get("summary") or {},
        "anomaly": anom,
    }


def cmd_runs(args):
    """收集一dir下的 report*.json（或显式 --inputs），做跨运行参数/指标 diff。"""
    paths = []
    if getattr(args, "dir", None):
        d = args.dir
        if not os.path.isdir(d):
            die("目录不存在: %s" % d)
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.lower().startswith("report") and f.lower().endswith(".json")]
    if getattr(args, "inputs", None):
        paths += [p.strip() for p in args.inputs.split(",") if p.strip()]
    if not paths:
        die("未找到任何 report*.json；请用 --dir 指定目录或 --inputs 指定逗号分隔的文件路径")
    paths = [os.path.abspath(p) for p in paths]
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)

    runs, ok = [], []
    for p in uniq:
        _, inner = _load_unwrap(p)
        if not isinstance(inner, dict) or "__error__" in inner:
            runs.append({"source": p, "error": inner.get("__error__", "无法解析") if isinstance(inner, dict) else "无法解析"})
            continue
        fld = _report_fields(inner)
        fld["source"] = p
        runs.append(fld); ok.append(fld)

    # 环境漂移：跨运行对比 numpy/scipy/sklearn/python
    env_keys = ["python_version", "numpy", "scipy", "sklearn"]
    env_drift = {}
    for k in env_keys:
        vals = {}
        for r in ok:
            v = r["environment"].get(k)
            if v is not None:
                vals[r.get("run_id") or r["source"]] = v
        if len(set(map(str, vals.values()))) > 1:
            env_drift[k] = vals

    # 形状变化：行/列/数值列数
    shape = [{"run_id": r.get("run_id") or r["source"], "rows": r["rows"],
              "n_columns": r["n_columns"], "n_numeric": r["n_numeric"]} for r in ok]
    shape_changed = len({(r["rows"], r["n_columns"], r["n_numeric"]) for r in ok}) > 1

    # 指标 diff：各 run 对每个数值列的 mean/std
    col_means = {}
    for r in ok:
        for c, stat in (r.get("summary") or {}).items():
            if isinstance(stat, dict) and stat.get("mean") is not None:
                col_means.setdefault(c, []).append(
                    {"run_id": r.get("run_id") or r["source"],
                     "mean": stat.get("mean"), "std": stat.get("std")})
    metric_diffs = {}
    for c, lst in col_means.items():
        means = [x["mean"] for x in lst]
        if len(means) >= 2 and any(abs(means[i] - means[0]) > 1e-9 for i in range(1, len(means))):
            metric_diffs[c] = lst

    # 异常 diff（仅在异常计数跨运行存在差异时保留，避免噪声）
    anom_raw = {}
    for r in ok:
        for c, a in (r.get("anomaly") or {}).items():
            anom_raw.setdefault(c, []).append(
                {"run_id": r.get("run_id") or r["source"],
                 "count": (a.get("count") if isinstance(a, dict) else a)})
    anom_diffs = {}
    for c, lst in anom_raw.items():
        counts = [x["count"] for x in lst]
        if len(counts) >= 2 and len(set(counts)) > 1:
            anom_diffs[c] = lst

    # 可复现性告警
    flags = []
    if env_drift:
        flags.append("运行环境不一致：%s 在不同运行中取值不同，可能影响结果可复现性" %
                     "、".join(env_drift.keys()))
    if shape_changed:
        flags.append("数据集形状（行数/列数）在不同运行间不一致，疑似输入数据或清洗步骤变化")
    if metric_diffs:
        flags.append("共 %d 个数值列的均值在运行间发生变化，建议核对数据/随机种子/预处理" % len(metric_diffs))

    result = {
        "n_runs": len(runs), "n_ok": len(ok),
        "runs": [{"run_id": r.get("run_id") or r["source"], "rows": r["rows"],
                  "n_columns": r["n_columns"], "n_numeric": r["n_numeric"],
                  "n_anomaly": r["n_anomaly"], "environment": r["environment"]} for r in ok],
        "environment_drift": env_drift,
        "shape_changes": {"changed": shape_changed, "detail": shape},
        "metric_diffs": metric_diffs,
        "anomaly_diffs": anom_diffs,
        "reproducibility_flags": flags,
    }
    emit({"status": "ok", "task": "runs", "result": result})


def _apa_p(p):
    """APA 风格 p 值：p < .001 / p = .032。"""
    if p is None:
        return None
    try:
        pf = float(p)
    except Exception:
        return None
    if pf != pf:
        return None
    if pf < 0.001:
        return "p < .001"
    return "p = %.3f" % pf


def _block_effect(d):
    """从结果块抽取效应量（Cohen's d / η² / ε² / ω / ICC / BF10）文本。"""
    if not isinstance(d, dict):
        return None
    cmap = [("cohen_d", "Cohen's d"), ("eta2", "η²"), ("epsilon_squared", "ε²"),
            ("omega2", "ω²"), ("bf10", "BF10")]
    bits = []
    for key, label in cmap:
        v = d.get(key)
        if isinstance(v, (int, float)) and v == v:
            bits.append("%s = %.3f" % (label, v))
    return "；".join(bits) if bits else None


def _human_label(path, task):
    m = {
        "anova": "方差分析", "ttest_independent": "独立样本 t 检验",
        "ttest_paired": "配对样本 t 检验", "ttest_one_sample": "单样本 t 检验",
        "chisq": "卡方检验", "correlation": "相关分析", "normality": "正态性检验",
        "tukey": "Tukey/Games-Howell 事后", "rm_anova": "重复测量 ANOVA",
        "ancova": "ANCOVA", "mixed_anova": "混合 ANOVA", "mediation": "中介分析",
        "dunnett": "Dunnett 事后", "nemenyi": "Nemenyi 事后", "scheffe": "Scheffe 事后",
        "wilcoxon": "Wilcoxon 符号秩检验", "friedman": "Friedman 检验",
        "mcnemar": "McNemar 检验", "mann_whitney": "Mann-Whitney U 检验",
        "moderation": "调节分析", "bootstrap_mediation": "Bootstrap 中介检验",
        "stepwise": "逐步回归",
        "icc": "ICC 组内相关", "cronbach": "Cronbach α 信度", "vif": "VIF 多重共线性",
        "fixed_effects": "固定效应", "variance_components": "方差组分",
    }
    if not path:
        return {"stats": "统计结果", "report": "溯源报告", "lmm": "线性混合模型",
                "factor": "因子分析"}.get(task, task or "结果")
    return m.get(path[-1], path[-1].replace("_", " "))


def _collect_interpretations(node, path, out):
    """递归收集所有含 interpretation 的块（记录键路径与其节点）。"""
    if isinstance(node, dict):
        if "interpretation" in node and isinstance(node["interpretation"], str):
            out.append((tuple(path), node))
        for k, v in node.items():
            if k == "interpretation":
                continue
            _collect_interpretations(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _collect_interpretations(v, path + ["[%d]" % i], out)


def _narrate_report(inner):
    """report 结果无 interpretation，依据 dataset/summary/anomaly 生成叙述。"""
    ds = inner.get("dataset") or {}
    cols = ds.get("columns") or []
    num = ds.get("numeric_columns") or []
    summary = inner.get("summary") or {}
    anom = inner.get("anomaly_iqr_L2") or {}
    n_anom = 0
    for a in anom.values():
        n_anom += (a.get("count") if isinstance(a, dict) else (a or 0))
    lines = []
    lines.append("本研究数据共包含 %d 条记录、%d 个变量（其中数值变量 %d 个）。"
                 % (ds.get("rows") or 0, len(cols), len(num)))
    parts = []
    for c in num[:8]:
        s = summary.get(c)
        if isinstance(s, dict) and s.get("mean") is not None:
            parts.append("%s（M=%.2f，SD=%.2f）" % (c, s["mean"], s.get("std") or 0))
    if parts:
        lines.append("主要数值变量的描述统计为：" + "；".join(parts) + "。")
    flagged = [c for c, v in anom.items()
               if (v.get("count") if isinstance(v, dict) else v)]
    if flagged:
        lines.append("基于 IQR（L2）规则的异常值检测共在 %d 处检出异常，涉及变量：%s。"
                     % (n_anom, "、".join(flagged)))
    else:
        lines.append("异常值检测（IQR L2）未检出明显异常。")
    disc = inner.get("discipline")
    if isinstance(disc, dict):
        lines.append("本报告按学科领域规则（%s）执行异常判定。"
                     % disc.get("name", disc.get("key", "")))
    return "\n".join(lines)


def _narrate_preamble(task):
    m = {
        "stats": "本研究采用多维统计分析方法对数据进行了系统的统计检验。",
        "lmm": "本研究采用线性混合模型（随机截距±斜率）分析数据。",
        "factor": "本研究采用探索性因子分析考察变量结构。",
        "report": "本研究数据集的溯源与质量特征如下。",
    }
    return m.get(task, "分析结果如下。")


def cmd_narrate(args):
    """读取 stats/report/lmm/factor 输出的 JSON，复用 interpretation 生成 APA 式段落草稿。"""
    task, inner = _load_unwrap(args.input)
    if not isinstance(inner, dict) or "__error__" in inner:
        die(inner.get("__error__", "无法加载输入") if isinstance(inner, dict) else "无法加载输入")

    if task == "report":
        draft = _narrate_report(inner)
        blocks = []
    else:
        raw = []
        _collect_interpretations(inner, [], raw)
        seen, uniq = set(), []
        for path, node in raw:
            if path in seen:
                continue
            seen.add(path); uniq.append((path, node))
        blocks = []
        for path, node in uniq:
            interp = node.get("interpretation") or ""
            if not interp:
                continue
            apa = _apa_p(node.get("p_value"))
            eff = _block_effect(node)
            sentence = interp.rstrip("。. ")
            extras = [x for x in (apa, eff) if x]
            if extras:
                sentence += "（" + "；".join(extras) + "）"
            sentence += "。"
            blocks.append({
                "key": ".".join(path) or (task or "result"),
                "label": _human_label(path, task),
                "interpretation": interp,
                "p_value": node.get("p_value"),
                "effect": eff,
                "sentence": sentence,
            })
        # lmm 固定效应补充（直接服务于 APA 段落，不重复计算）
        if task == "lmm":
            fe = inner.get("fixed_effects") or []
            fe_bits = []
            for c in fe:
                if isinstance(c, dict):
                    coef = c.get("coef")
                    fe_bits.append("%s（β=%s，%s）" % (
                        c.get("term"),
                        ("%.3f" % coef) if isinstance(coef, (int, float)) else coef,
                        _apa_p(c.get("p_value")) or "p=NA"))
            if fe_bits:
                blocks.append({"key": "fixed_effects", "label": "固定效应",
                               "interpretation": "", "p_value": None, "effect": None,
                               "sentence": "固定效应估计：" + "；".join(fe_bits) + "。"})
        mc = inner.get("multiple_comparison") if isinstance(inner, dict) else None
        draft = _narrate_preamble(task)
        if blocks:
            draft += "\n\n"
            if len(blocks) > 1:
                for i, b in enumerate(blocks, 1):
                    lab = ("【%s】" % b["label"]) if b["label"] else ""
                    draft += "%d. %s%s\n" % (i, lab, b["sentence"])
            else:
                draft += blocks[0]["sentence"] + "\n"
        if isinstance(mc, dict) and mc.get("note"):
            draft += "\n" + mc["note"]
    emit({"status": "ok", "task": "narrate",
          "result": {"source_task": task, "n_blocks": len(blocks),
                     "blocks": blocks, "draft": draft}})


def cmd_guide(args):
    """新手引导：根据分析目标推荐合适的子命令。"""
    guide_text = """BaiChuanShuHui 数据分析内核 — 新手引导
==========================================

■ 第一步：加载数据并了解概况
  load --input data.csv
  stats --input data.csv --describe --normality

■ 第二步：根据分析目标选择方法

  ▸ 比较两组差异（独立样本）
    stats --input data.csv --ttest 因子,数值        # 参数 t 检验（自动 Levene 门控）
    stats --input data.csv --mannwhitney 因子,数值   # 非参 Mann-Whitney U

  ▸ 比较两组差异（配对/重复测量）
    stats --input data.csv --ttest-paired 字段1,字段2  # 配对 t 检验
    stats --input data.csv --wilcoxon 字段1,字段2      # 非参 Wilcoxon 符号秩

  ▸ 比较多组差异（≥3 组）
    stats --input data.csv --anova 因子,数值         # 单/双因素 ANOVA（III 型 SS）
    stats --input data.csv --tukey 因子,数值          # 事后多重比较

  ▸ 重复测量设计
    stats --input data.csv --rm-anova 受试者,条件,数值 # 重复测量 ANOVA + 球形检验
    stats --input data.csv --friedman 受试者,条件,数值 # 非参 Friedman + Dunn

  ▸ 变量关系与预测
    stats --input data.csv --corr                    # 相关矩阵 + p 值 + FDR
    regress --input data.csv --y Y --x X1,X2,X3      # 多元回归 + HC 稳健 SE
    stats --input data.csv --stepwise y,x1,x2,x3     # 逐步回归

  ▸ 中介与调节
    stats --input data.csv --mediation x,m,y         # Baron-Kenny + Sobel
    stats --input data.csv --bootstrap-mediation x,m,y  # Bootstrap CI
    stats --input data.css --moderation x,w,y        # 交互项 + 简单斜率

  ▸ 分类数据
    stats --input data.csv --chisq 字段A,字段B       # 卡方独立性
    stats --input data.csv --mcnemar 字段1,字段2     # 配对二分类

  ▸ 信效度
    factor --input data.csv --features v1,v2,v3 --mode reliability  # McDonald's ω
    stats --input data.csv --cronbach items           # Cronbach's α

  ▸ 混合模型与层次结构
    lmm --input data.csv --y Y --x X --group subject  # 线性混合模型
    stats --input data.csv --mixed within,between,subject,value  # 混合 ANOVA

  ▸ 数据质量
    missing --input data.csv                          # 缺失值分析
    anomaly --input data.csv                          # 异常值检测
    quality --input data.csv                          # 数据质量评分

■ 第三步：生成报告
  report --input data.csv --desc --anomaly
  narrate --input stats_output.json                  # APA 式结果草稿

■ 全局选项
  --quiet        静默模式（省略溯源信息）
  --verbose      显示进度（bootstrap/置换检验）
  --out-format   输出格式 json/csv/md/html
  --version      查看版本

■ 提示：每个子命令加 --help 查看完整参数列表"""
    emit({"status": "ok", "task": "guide", "result": {"guide": guide_text}})


def cmd_preregister(args):
    """预注册硬锁：记录分析方案的哈希，后续运行时验证一致性。"""
    import hashlib as _hl
    plan = {
        "input": args.input,
        "command": args.command,
        "alpha": float(args.alpha) if args.alpha else 0.05,
        "ss_type": getattr(args, "ss_type", "III"),
        "adjust": getattr(args, "adjust", None),
    }
    plan_str = json.dumps(plan, sort_keys=True, ensure_ascii=False)
    plan_hash = _hl.sha256(plan_str.encode("utf-8")).hexdigest()
    prereg = {
        "plan": plan,
        "plan_hash": plan_hash,
        "kernel_version": KERNEL_VERSION,
        "created_at": None,
    }
    try:
        import datetime
        prereg["created_at"] = datetime.datetime.now().isoformat()
    except Exception:
        pass
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(prereg, f, ensure_ascii=False, indent=2)
        emit({"status": "ok", "task": "preregister",
              "result": {"output_file": args.output, "plan_hash": plan_hash,
                         "plan": plan,
                         "interpretation": "预注册方案已保存至 %s。后续运行时用 --check-preregister %s 验证一致性。" % (args.output, args.output)}})
    else:
        emit({"status": "ok", "task": "preregister",
              "result": prereg})


def _check_preregister(prereg_path, current_plan):
    """验证当前分析方案是否与预注册方案一致。"""
    try:
        with open(prereg_path, "r", encoding="utf-8") as f:
            prereg = json.load(f)
    except Exception as e:
        _warn("预注册文件读取失败: %s" % e)
        return
    import hashlib as _hl
    plan_str = json.dumps(current_plan, sort_keys=True, ensure_ascii=False)
    current_hash = _hl.sha256(plan_str.encode("utf-8")).hexdigest()
    locked_hash = prereg.get("plan_hash")
    if locked_hash == current_hash:
        _warn("✓ 预注册验证通过：当前分析方案与预注册方案一致")
    else:
        _warn("⚠ 预注册不一致：当前分析方案与预注册方案不匹配！"
              "预注册方案：%s；当前方案：%s" % (prereg.get("plan"), current_plan))


def build_parser():
    p = argparse.ArgumentParser(description="BaiChuanShuHui AI4SS 数据处理内核")
    p.add_argument("--version", action="version", version="BaiChuanShuHui kernel v%s" % KERNEL_VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("load", help="加载并预览数据")
    pl.add_argument("--input", required=True)
    pl.add_argument("--format", default="auto")
    pl.add_argument("--max-rows", type=int, default=None)
    pl.set_defaults(func=cmd_load)

    pc = sub.add_parser("clean", help="数据清洗")
    pc.add_argument("--input", required=True)
    pc.add_argument("--drop-empty-cols", action="store_true")
    pc.add_argument("--missing", default="drop", choices=["drop", "mean", "median", "ffill"])
    pc.add_argument("--output", default=None)
    pc.set_defaults(func=cmd_clean)

    pa = sub.add_parser("anomaly", help="异常值检测/修复")
    pa.add_argument("--input", required=True)
    pa.add_argument("--method", default="iqr",
                    choices=["iqr", "zscore", "mad", "isolation_forest", "mahalanobis"])
    pa.add_argument("--grade", default="L2", choices=["L1", "L2", "L3"])
    pa.add_argument("--field", default=None,
                    help="待检测字段（逗号分隔，可选）；缺省自动排除标识列(id/编号)")
    pa.add_argument("--repair", action="store_true")
    pa.add_argument("--output", default=None)
    pa.add_argument("--contamination", type=float, default=None,
                    help="isolation_forest 污染率（覆盖 grade 默认：L1 0.10/L2 0.05/L3 0.01）")
    pa.add_argument("--discipline", default=None,
                    help="学科模式：biomed/agri_env/materials/cs/chem/civil_geo/econ_soc/math_stat/general"
                         "/通用别名/auto；按学科领域规则(method+factor+grade+advice)复用 _anomaly_column 检测，"
                         "未指定则保持原行为")
    pa.set_defaults(func=cmd_anomaly)

    ps = sub.add_parser("stats", help="多维统计分析")
    ps.add_argument("--input", required=True)
    ps.add_argument("--describe", action="store_true")
    ps.add_argument("--corr", action="store_true")
    ps.add_argument("--normality", action="store_true")
    ps.add_argument("--anova", default=None, help="A,B,value 或 A,value（自动串联正态性+齐性门控）")
    ps.add_argument("--ss-type", default="III", choices=["I", "III"],
                    help="双因素 ANOVA 平方和类型：III（默认）/ I（序贯）")
    ps.add_argument("--ttest-one", dest="ttest_one", default=None,
                    help="字段,参考值 —— 单样本 t 检验")
    ps.add_argument("--ttest", default=None,
                    help="因子,数值 —— 两独立样本 t（Levene 门控自动路由 Student/Welch）")
    ps.add_argument("--ttest-paired", dest="ttest_paired", default=None,
                    help="字段1,字段2 —— 配对样本 t 检验")
    ps.add_argument("--mannwhitney", default=None,
                    help="因子,数值 —— 两独立样本 Mann-Whitney U 检验（非参，附 Cliff's δ / rank-biserial 效应量）")
    ps.add_argument("--wilcoxon", default=None,
                    help="字段1,字段2 —— 配对样本 Wilcoxon 符号秩检验（非参，z 近似+结校正，效应量 r）")
    ps.add_argument("--friedman", default=None,
                    help="受试者,条件,数值 —— Friedman 检验（重复测量非参，Kendall's W + Dunn 事后）")
    ps.add_argument("--mcnemar", default=None,
                    help="字段1,字段2 —— McNemar 检验（配对二分类，连续性校正+小样本精确二项回退）")
    ps.add_argument("--chisq", default=None,
                    help="字段A,字段B（列联表独立性）或 字段（均匀拟合优度）")
    ps.add_argument("--tukey", default=None, help="factor,value（方差齐用 Tukey HSD，不齐自动 Games-Howell）")
    ps.add_argument("--vif", action="store_true")
    ps.add_argument("--rm-anova", dest="rm_anova", default=None,
                    help="重复测量（组内）ANOVA：'受试者,组内因子,数值'（长格式）；含球形检验与 GG 校正")
    ps.add_argument("--alpha", type=float, default=None,
                    help="显著性水平（默认 0.05；影响显著性判定/齐性门控/结论文本）")
    ps.add_argument("--shapiro-max-n", dest="shapiro_max_n", type=int, default=None,
                    help="Shapiro-Wilk 截断上限（默认 5000）")
    ps.add_argument("--adjust", default=None, choices=["holm", "bonferroni", "fdr"],
                    help="多检验 p 值多重比较校正：holm / bonferroni / fdr（默认不校正，仅提示）")
    ps.add_argument("--permutation", action="store_true",
                    help="对相关/独立 t 检验额外给出置换检验 p 值（精确经验显著性，零依赖）")
    ps.add_argument("--n-perm", dest="n_perm", type=int, default=2000, help="置换检验重采样次数（默认 2000）")
    ps.add_argument("--seed", type=int, default=0, help="置换检验随机种子（默认 0，保证可复现）")
    ps.add_argument("--dunnett", default=None, help="单因素事后 Dunnett（多对一）：'factor,value'（需配 --control）")
    ps.add_argument("--control", default=None, help="Dunnett 对照水平（指定 --dunnett 时生效）")
    ps.add_argument("--nemenyi", default=None, help="Kruskal-Wallis 事后 Nemenyi：'factor,value'")
    ps.add_argument("--scheffe", default=None, help="单因素事后 Scheffe：'factor,value'")
    ps.add_argument("--compact-letters", dest="compact_letters", default=None,
                    help="对 --anova 分组结果做字母标记法(CLD)：'factor,value'")
    ps.add_argument("--icc", default=None, help="组内相关 ICC：'subject,rater,value'（长格式）")
    ps.add_argument("--icc-model", dest="icc_model", default="2-1", choices=["1-1", "2-1", "3-1"],
                    help="ICC 模型：1-1 / 2-1 / 3-1（默认 2-1）")
    ps.add_argument("--cronbach", default=None, help="Cronbach α 信度：'item1,item2,...'（多题项列）")
    ps.add_argument("--ancova", default=None, help="ANCOVA：'group,covariate,value'（单因素+协变量校正）")
    ps.add_argument("--mixed", default=None, help="混合 ANOVA：'within,between,subject,value'")
    ps.add_argument("--mediation", default=None, help="中介分析：'x,m,y'（Baron-Kenny + Sobel）")
    ps.add_argument("--moderation", default=None,
                    help="调节分析：'x,w,y'（中心化交互项 + 简单斜率 + Johnson-Neyman）")
    ps.add_argument("--bootstrap-mediation", dest="bootstrap_mediation", default=None,
                    help="Bootstrap 中介检验：'x,m,y'（BCa CI for indirect effect，Hayes PROCESS 式）")
    ps.add_argument("--n-boot", dest="n_boot", type=int, default=5000,
                    help="Bootstrap 中介检验重采样次数（默认 5000）")
    ps.add_argument("--stepwise", default=None,
                    help="逐步回归：'y,x1,x2,...'（配合 --direction forward/backward/best_subset）")
    ps.add_argument("--direction", default="forward", choices=["forward", "backward", "best_subset"],
                    help="逐步回归方向：forward（默认）/ backward / best_subset")
    ps.add_argument("--entry-p", dest="entry_p", type=float, default=0.05,
                    help="逐步回归进入/剔除阈值（默认 0.05）")
    ps.add_argument("--check-preregister", dest="check_preregister", default=None,
                    help="预注册验证：加载预注册 JSON 文件，验证当前分析方案是否一致")
    ps.set_defaults(func=cmd_stats)

    pm = sub.add_parser("missing", help="缺失值与哨兵值专项分析")
    pm.add_argument("--input", required=True)
    pm.add_argument("--sample-rows", type=int, default=60, help="缺失矩阵抽样行数")
    pm.add_argument("--fix-sentinel", action="store_true", help="把哨兵值(-999等)视为缺失并处理")
    pm.add_argument("--strategy", default="blank", choices=["blank", "mean", "median", "mice"],
                    help="缺失/哨兵值处理策略：置空/均值/中位数/多重插补(MICE)")
    pm.add_argument("--sentinels", default=None,
                    help="自定义哨兵值列表（逗号分隔，如 -999,-99,9999；缺省用内置默认表）")
    pm.add_argument("--seed", type=int, default=0, help="MICE 插补随机种子（默认 0，保证可复现）")
    pm.add_argument("--output", default=None)
    pm.set_defaults(func=cmd_missing)

    pq = sub.add_parser("quality", help="数据质量评分（完整度/异常/分布/共线性）")
    pq.add_argument("--input", required=True)
    pq.add_argument("--alpha", type=float, default=None, help="正态性判定显著性水平（默认 0.05）")
    pq.add_argument("--shapiro-max-n", dest="shapiro_max_n", type=int, default=None,
                    help="Shapiro-Wilk 截断上限（默认 5000）")
    pq.set_defaults(func=cmd_quality)

    pk = sub.add_parser("skew", help="偏度诊断与正态化变换")
    pk.add_argument("--input", required=True)
    pk.add_argument("--field", required=True, help="待变换的数值字段")
    pk.add_argument("--method", default="log", choices=["log", "sqrt", "boxcox", "johnson"])
    pk.add_argument("--lambda", dest="lmbda", type=float, default=None, help="Box-Cox λ（缺省自动估计）")
    pk.add_argument("--output", default=None, help="写出含新列的 CSV")
    pk.add_argument("--alpha", type=float, default=None, help="正态性判定显著性水平（默认 0.05）")
    pk.add_argument("--shapiro-max-n", dest="shapiro_max_n", type=int, default=None,
                    help="Shapiro-Wilk 截断上限（默认 5000）")
    pk.set_defaults(func=cmd_skew)

    pr = sub.add_parser("report", help="生成溯源报告")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default=None)
    pr.add_argument("--discipline", default=None, help="同 anomaly：按学科领域规则生成溯源报告中的异常判定")
    pr.set_defaults(func=cmd_report)

    pv = sub.add_parser("viz", help="生成可视化图表(自包含ECharts HTML)")
    pv.add_argument("--input", required=True)
    pv.add_argument("--kind", default=None, choices=VIZ_KINDS,
                    help="图表类型（可选；省略且给出 --discipline 时自动选用学科首图）: " + ", ".join(VIZ_KINDS))
    pv.add_argument("--x", default=None)
    pv.add_argument("--y", default=None)
    pv.add_argument("--z", default=None)
    pv.add_argument("--output", default=None)
    pv.add_argument("--title", default=None)
    pv.add_argument("--theme", default="light", choices=["light", "dark"])
    pv.add_argument("--bins", type=int, default=20)
    pv.add_argument("--ring", action="store_true", help="饼图改为环形")
    pv.add_argument("--discipline", default=None, help="学科模式（同 anomaly）；推断后输出 recommended_kinds 推荐图表")
    pv.set_defaults(func=cmd_viz)

    pd = sub.add_parser("discipline", help="九大学科自动推断 / 领域规则与图表模板清单")
    pd.add_argument("--input", default=None, help="待推断学科的数据文件（与 --list 二选一）")
    pd.add_argument("--list", action="store_true", help="列出全部九大学科及其领域规则、图表模板")
    pd.set_defaults(func=cmd_discipline)

    pt = sub.add_parser("tost", help="TOST 等价/非劣效检验（证明等效而非仅不显著）")
    pt.add_argument("--input", required=True)
    pt.add_argument("--ttest", default=None, help="因子,数值（两独立样本均值 TOST）")
    pt.add_argument("--paired", default=None, help="字段1,字段2（配对均值 TOST）")
    pt.add_argument("--anova", default=None, help="因子,数值（单因素 ANOVA 等价：全部组对 TOST）")
    pt.add_argument("--prop", default=None, help="因子,取值(0/1)（两比例 TOST，正态近似）")
    pt.add_argument("--margin", required=True, help="等价界 Δ（正数值；比例/均值差与界值比较）")
    pt.add_argument("--direction", default="equivalence", choices=["equivalence", "superiority", "non_inferiority"],
                   help="检验方向：equivalence（双向 TOST，默认）/ superiority（单侧 diff>+Δ）/ non_inferiority（单侧 diff>−Δ）")
    pt.add_argument("--alpha", type=float, default=None, help="显著性水平（默认 0.05）")
    pt.set_defaults(func=cmd_tost)

    pp = sub.add_parser("power", help="统计功效 / 所需样本量分析")
    pp.add_argument("--test", required=True, choices=["t", "corr", "prop", "anova", "tost", "regression", "survival", "manova"])
    pp.add_argument("--alpha", type=float, default=None, help="显著性水平（默认 0.05）")
    pp.add_argument("--power", default=None, help="目标功效（0-1）；与 --n 二选一求解")
    pp.add_argument("--n", default=None, help="每组样本量（survival/manova/regression 为总样本量）")
    pp.add_argument("--d", default=None, help="t 检验均值差")
    pp.add_argument("--sd", default=None, help="t 检验合并标准差（默认 1）；tost 亦用于等价界计算")
    pp.add_argument("--alternative", default="two-sided", choices=["two-sided", "one-sided"])
    pp.add_argument("--r", default=None, help="corr: 相关系数")
    pp.add_argument("--p1", default=None, help="prop: 组1比例")
    pp.add_argument("--p2", default=None, help="prop: 组2比例")
    pp.add_argument("--k", default=None, help="anova: 组数；manova: 组数")
    pp.add_argument("--f", default=None, help="anova/manova: 效应量 f")
    pp.add_argument("--eta2", default=None, help="anova/manova: η²（自动转 f）")
    pp.add_argument("--margin", default=None, help="tost: 等价界 Δ（非负；所需样本量分析必填）")
    pp.add_argument("--r2", default=None, help="regression: 决定系数 R²（0-1）")
    pp.add_argument("--p", default=None, help="manova: 因变量个数 p")
    pp.add_argument("--hr", default=None, help="survival: 风险比 HR")
    pp.add_argument("--p-event", dest="p_event", default=None, help="survival: 对照组事件概率（0-1）")
    pp.set_defaults(func=cmd_power)

    prb = sub.add_parser("robust", help="稳健统计量（截尾均值 / Huber M 估计）")
    prb.add_argument("--input", required=True)
    prb.add_argument("--field", required=True, help="数值字段")
    prb.add_argument("--trim", type=float, default=0.2, help="截尾比例（默认 0.2）")
    prb.add_argument("--k", type=float, default=1.345, help="Huber 调节常数（默认 1.345）")
    prb.set_defaults(func=cmd_robust)

    pb = sub.add_parser("bootstrap", help="Bootstrap 百分位置信区间")
    pb.add_argument("--input", required=True)
    pb.add_argument("--field", required=True, help="数值字段")
    pb.add_argument("--stat", default="mean", choices=["mean", "median", "sd"])
    pb.add_argument("--n", type=int, default=2000, help="重采样次数")
    pb.add_argument("--seed", type=int, default=0, help="随机种子（默认 0，保证可复现）")
    pb.add_argument("--alpha", type=float, default=0.05)
    pb.add_argument("--method", default="percentile", choices=["percentile", "bca"],
                   help="置信区间方法：percentile(默认) / bca(偏差校正加速，偏态分布更准)")
    pb.set_defaults(func=cmd_bootstrap)

    preg = sub.add_parser("regress", help="回归（线性/Logistic/Poisson，含稳健标准误与效应量）")
    preg.add_argument("--input", required=True)
    preg.add_argument("--y", required=True, help="因变量字段")
    preg.add_argument("--x", required=True, help="自变量字段，逗号分隔")
    preg.add_argument("--family", default="gaussian", choices=["gaussian", "logistic", "poisson"],
                      help="模型族：gaussian(默认,OLS+HC3) / logistic(二分类,OR+95%%CI) / poisson(计数,IRR+95%%CI)")
    preg.add_argument("--hc", type=int, default=3, choices=[0, 1, 2, 3],
                     help="异方差稳健标准误类型（仅 gaussian）：0=White(HC0) / 1=HC1(df校正) / 2=HC2 / 3=HC3(默认)")
    preg.set_defaults(func=cmd_regress)

    psens = sub.add_parser("sensitivity", help="稳健性/敏感性分析（剔除 L3 离群点前后结论一致性）")
    psens.add_argument("--input", required=True)
    psens.add_argument("--ttest", default=None, help="因子,数值（对独立样本 t 检验做敏感性分析）")
    psens.add_argument("--anova", default=None, help="因子,数值（对单因素 ANOVA 做敏感性分析）")
    psens.set_defaults(func=cmd_sensitivity)

    pmeta = sub.add_parser("meta", help="元分析（合并效应量/异质性 I²/森林图数据）")
    pmeta.add_argument("--input", required=True)
    pmeta.add_argument("--mode", default="es", choices=["es", "cont", "or2x2", "rr2x2"],
                       help="输入形态：es(研究级效应量) / cont(连续结局 n,m,sd) / or2x2 / rr2x2")
    pmeta.add_argument("--spec", default=None,
                       help="列名映射，逗号分隔。es: study,es,se(或 study,es,ci_low,ci_high)；"
                            "cont: study,n1,n2,m1,m2,sd1,sd2；2x2: study,a,b,c,d")
    pmeta.add_argument("--alpha", type=float, default=None, help="显著性/置信水平（默认 0.05）")
    pmeta.add_argument("--estimator", default="dl", choices=["dl", "reml", "hs", "fixed"],
                       help="随机效应 τ² 估计：dl(DerSimonian-Laird,默认) / reml / hs(Hunter-Schmidt) / fixed(仅固定效应)")
    pmeta.add_argument("--output", default=None, help="导出结果（.json 或 .html 森林图）")
    pmeta.set_defaults(func=cmd_meta)

    ppred = sub.add_parser("predict", help="预测建模（分类/回归，含数据泄露门控）")
    ppred.add_argument("--input", required=True)
    ppred.add_argument("--target", required=True, help="目标变量列名")
    ppred.add_argument("--features", required=True, help="特征列名，逗号分隔")
    ppred.add_argument("--task", default=None, choices=["regression", "classification"],
                       help="任务类型（缺省按目标列数据类型自动判定）")
    ppred.add_argument("--model", default="auto", choices=["auto", "linear", "logistic", "knn"],
                       help="模型：auto(回归→linear/分类→logistic) / linear / logistic / knn")
    ppred.add_argument("--split", type=float, default=0.25, help="验证集比例（默认 0.25）")
    ppred.add_argument("--k", type=int, default=5, help="knn 近邻数（默认 5）")
    ppred.add_argument("--seed", type=int, default=0, help="随机种子（默认 0，保证可复现）")
    ppred.add_argument("--cv", type=int, default=0, help="k 折分层交叉验证（≥2 时启用；每折训练集内标准化/SMOTE，无泄露）")
    ppred.add_argument("--compare", action="store_true", help="多模型排名（回归: linear/knn；分类: logistic/knn）")
    ppred.add_argument("--importance", action="store_true", help="置换特征重要性（5 折重排，纯依赖）")
    ppred.add_argument("--smote", action="store_true", help="分类时对少数类过采样（仅训练集，防泄露）")
    ppred.add_argument("--save-pipeline", dest="save_pipeline", default=None,
                       help="把编码/特征/模型配置保存为 JSON pipeline（可复用）")
    ppred.set_defaults(func=cmd_predict)

    pt1 = sub.add_parser("table1", help="论文 Table 1 基线特征表（分组描述/p值/SMD）")
    pt1.add_argument("--input", required=True)
    pt1.add_argument("--group", default=None, help="分组列（省略则整体描述）")
    pt1.add_argument("--vars", required=True, help="纳入变量列名，逗号分隔")
    pt1.add_argument("--output", default=None, help="导出 .csv/.md/.json")
    pt1.set_defaults(func=cmd_table1)

    psv = sub.add_parser("survival", help="生存分析（log-rank / Cox 比例风险回归）")
    psv.add_argument("--input", required=True)
    psv.add_argument("--mode", default="cox", choices=["logrank", "cox"], help="log-rank 组间检验 / Cox PH 回归")
    psv.add_argument("--time", required=True, help="生存时间列（数值）")
    psv.add_argument("--event", required=True, help="事件指示列（1=事件, 0/其他=删失）")
    psv.add_argument("--group", default=None, help="log-rank 分组列")
    psv.add_argument("--features", default=None, help="Cox 特征列，逗号分隔（数值）")
    psv.add_argument("--alpha", type=float, default=0.05)
    psv.add_argument("--output", default=None, help="Cox 结果导出 .json")
    psv.set_defaults(func=cmd_survival)

    ppf = sub.add_parser("profile", help="自动化 EDA（类型推断/Alerts 告警/关联矩阵/漂移对比）")
    ppf.add_argument("--input", required=True)
    ppf.add_argument("--compare", default=None, help="对比数据集路径，输出 train-test 漂移(PSI)")
    ppf.add_argument("--output", default=None, help="导出 .json")
    ppf.set_defaults(func=cmd_profile)

    pvld = sub.add_parser("validate", help="数据质量契约校验（Data Docs 式报告）")
    pvld.add_argument("--input", required=True)
    pvld.add_argument("--rules", default=None, help="规则文件(.json 列表)：{column,check,min,max,values}")
    pvld.add_argument("--rule", action="append", default=[],
                      help="内联规则 column:check:min:max（可重复；check∈not_null_rate/min_max/unique/allowed/dtype）")
    pvld.set_defaults(func=cmd_validate)

    pf = sub.add_parser("factor", help="探索性因子分析(EFA) + McDonald's ω 组合信度")
    pf.add_argument("--input", required=True)
    pf.add_argument("--features", required=True, help="参与分析的变量列，逗号分隔（≥3）")
    pf.add_argument("--mode", default="efa", choices=["efa", "reliability", "cfa"],
                   help="efa(默认,因子结构) / reliability(单因子解估计 McDonald's ω) / cfa(验证性因子分析,需 --cfa-model)")
    pf.add_argument("--n-factors", dest="n_factors", type=int, default=0,
                   help="提取因子数（默认按特征值>1 自动；reliability 模式强制单因子）")
    pf.add_argument("--rotation", default="varimax", choices=["varimax", "none"])
    pf.add_argument("--cfa-model", dest="cfa_model", default=None,
                   help="CFA 模型规格：'F1:i1,i2,i3;F2:i4,i5,i6'（变量名须来自 --features）")
    pf.add_argument("--cfa-orthogonal", dest="cfa_orthogonal", action="store_true",
                   help="CFA 因子正交（Φ=I）；缺省为斜交（估计因子相关）")
    pf.set_defaults(func=cmd_factor)

    pl = sub.add_parser("lmm", help="线性混合模型（随机截距±随机斜率，MME+EM）")
    pl.add_argument("--input", required=True)
    pl.add_argument("--y", required=True, help="因变量字段")
    pl.add_argument("--x", required=True, help="固定效应自变量，逗号分隔")
    pl.add_argument("--group", required=True, help="随机截距分组列（如受试者/班级）")
    pl.add_argument("--random-slope", dest="random_slope", default=None,
                   help="随机斜率列（可选；与 --group 配套，每个组一条斜率）")
    pl.set_defaults(func=cmd_lmm)

    prn = sub.add_parser("runs", help="跨运行溯源报告对比（收集 report*.json 做 diff）")
    prn.add_argument("--dir", default=".", help="含 report*.json 的目录（默认当前目录）")
    prn.add_argument("--inputs", default=None,
                     help="显式指定逗号分隔的 report JSON 路径（与 --dir 合并）")
    prn.set_defaults(func=cmd_runs)

    pnr = sub.add_parser("narrate", help="由 stats/report JSON 生成 APA 式结果草稿")
    pnr.add_argument("--input", required=True, help="stats / report / lmm / factor 输出的 JSON 文件")
    pnr.set_defaults(func=cmd_narrate)

    pg = sub.add_parser("guide", help="新手引导：根据分析目标推荐子命令")
    pg.set_defaults(func=cmd_guide)

    ppreg = sub.add_parser("preregister", help="预注册硬锁：记录分析方案哈希，防止事后选择性报告")
    ppreg.add_argument("--input", required=True, help="数据文件路径")
    ppreg.add_argument("--command", required=True,
                       help="计划执行的分析命令（如 'stats --anova 组别,成绩'）")
    ppreg.add_argument("--alpha", default=None, type=float, help="显著性水平")
    ppreg.add_argument("--ss-type", dest="ss_type", default="III", choices=["I", "III"])
    ppreg.add_argument("--adjust", default=None, choices=["holm", "bonferroni", "fdr"])
    ppreg.add_argument("--output", default=None, help="输出预注册 JSON 文件路径")
    ppreg.set_defaults(func=cmd_preregister)

    psem = sub.add_parser("sem", help="结构方程模型 / 路径分析（含 CFI/TLI/RMSEA/SRMR 拟合指数）")
    psem.add_argument("--input", required=True, help="数据文件")
    psem.add_argument("--model", default=None,
                      help="模型方程串，分号分隔：'y ~ x1 + x2; F := i1 + i2 + i3'")
    psem.add_argument("--model-file", dest="model_file", default=None,
                      help="模型方程文件（.txt/.md），内容与 --model 同格式")
    psem.add_argument("--alpha", type=float, default=None, help="显著性水平（默认 0.05）")
    psem.set_defaults(func=cmd_sem)

    pirt = sub.add_parser("irt", help="项目反应理论（1PL/2PL/3PL + GRM/PCM 多分类，EM 边际极大似然）")
    pirt.add_argument("--input", required=True, help="数据文件（0/1 或分类整数计分）")
    pirt.add_argument("--items", required=True, help="题目（列）名，逗号分隔（0/1 或分类整数计分）")
    pirt.add_argument("--model-type", dest="model_type", default="2pl",
                      choices=["1pl", "2pl", "3pl", "grm", "pcm"],
                      help="模型：1pl(Rasch) / 2pl(默认) / 3pl(含猜测c) / grm(等级反应,分类计分) / pcm(部分计分,分类计分)")
    pirt.set_defaults(func=cmd_irt)

    # 全子命令可用的全局开关
    for sp in sub.choices.values():
        try:
            sp.add_argument("--reject-inf", dest="reject_inf", action="store_true",
                            help="严格模式：将 inf 等非有限数值按缺失处理（默认原值放行并告警）")
        except argparse.ArgumentError:
            pass
        try:
            sp.add_argument("--quiet", dest="quiet", action="store_true",
                            help="静默模式：仅输出结果，省略 warnings / provenance")
        except argparse.ArgumentError:
            pass
        try:
            sp.add_argument("--verbose", dest="verbose", action="store_true",
                            help="详细模式：输出进度信息到 stderr（bootstrap / 置换检验等大数据量时有用）")
        except argparse.ArgumentError:
            pass
        try:
            sp.add_argument("--out-format", dest="out_format", default="json",
                            choices=["json", "csv", "md", "html"],
                            help="输出格式：json（默认）/ csv / md / html")
        except argparse.ArgumentError:
            pass
    return p


def _batch_expand(inp):
    """若 --input 为目录或 glob 且命中多个文件，返回文件列表；否则空列表。"""
    if not inp:
        return []
    if os.path.isdir(inp):
        files = [os.path.join(inp, f) for f in sorted(os.listdir(inp))
                 if f.lower().endswith((".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx"))]
        return files
    if any(ch in inp for ch in "*?["):
        hits = sorted(glob.glob(inp))
        if hits:
            return hits
    return []


def _run_batch(args, files):
    """批量模式：对列表中的每个文件，以相同子命令/参数重跑一次，汇总 JSON。"""
    argv = sys.argv[1:]
    subcommand = argv[0] if argv else "unknown"
    results = []
    n_ok = 0
    for fp in files:
        sub = [subcommand]
    skip = False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--output":
            skip = True
            continue
        if a.startswith("--output="):
            continue
        sub.append(fp if a == args.input else a)
        try:
            proc = subprocess.run([sys.executable, os.path.abspath(__file__)] + sub,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=300)
        except subprocess.TimeoutExpired:
            results.append({"input": fp, "status": "error", "message": "超时"})
            continue
        try:
            out = json.loads(proc.stdout.decode("utf-8", "replace"))
            if out.get("status") == "ok":
                n_ok += 1
        except Exception:
            out = {"status": "error", "message": proc.stderr.decode("utf-8", "replace")[:300]}
        results.append({"input": fp, "output": out})
    emit({"status": "ok", "task": "batch", "subcommand": subcommand,
          "n_files": len(files), "n_ok": n_ok, "results": results})


def main():
    global _VERBOSITY, _OUTPUT_FORMAT
    args = build_parser().parse_args()
    if getattr(args, "reject_inf", False):
        _REJECT_INF["on"] = True
    if getattr(args, "verbose", False):
        _VERBOSITY = "verbose"
    elif getattr(args, "quiet", False):
        _VERBOSITY = "quiet"
    _of = getattr(args, "out_format", "json")
    if _of and _of != "json":
        _OUTPUT_FORMAT = _of
    _PROV["param_hash"] = _compute_param_hash(args)
    inp = getattr(args, "input", None)
    batch_files = _batch_expand(inp) if inp else []
    if len(batch_files) > 1:
        _run_batch(args, batch_files)
        return
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        die("运行错误: %s" % e)


if __name__ == "__main__":
    main()
