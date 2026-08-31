"""macOS 本地评测包装：直接调用 bigcodebench.evaluate.evaluate。

在 bigcode-venv（research/.benchmark_runtime/bigcode-venv）的解释器下运行，
绕开 fire CLI 在本机的两个坑（详见 bigcode_track.py 模块 docstring）：

1. fire 把 ``--pass_k 1`` 解析成 int，evaluate 内部 ``for k in passk``
   直接 TypeError —— 这里强制传字符串 "1"；
2. Darwin arm64 的 setrlimit(RLIMIT_AS/RLIMIT_DATA) 任何值都抛
   "current limit exceeds maximum limit" —— 传 0 让 reliability_guard
   整体跳过（评测机是自己的 benchmark 工作机，可接受无 rlimit）。

用法（bigcode-venv python）::

    $BCB_VENV/bin/python -m research.benchmark.bigcode_eval_local \
        <samples.jsonl> <override.jsonl|-> <parallel> <comma-joined-ids>

第二个参数为 "-" 时沿用已导出的 BIGCODEBENCH_OVERRIDE_PATH。
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    samples, override, parallel, selective = argv[1], argv[2], int(argv[3]), argv[4]
    if override != "-":
        os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = override

    from bigcodebench.evaluate import evaluate

    evaluate(
        split="instruct",
        subset="full",
        samples=samples,
        execution="local",
        parallel=parallel,
        pass_k="1",  # 必须是字符串，内部才 split(",")
        max_as_limit=0,
        max_data_limit=0,
        selective_evaluate=selective,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
