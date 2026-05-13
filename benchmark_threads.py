#!/usr/bin/env python3
"""Benchmark: 1/4/8/16/32 threads on xinference (port 9997)."""

import json
import os
import re
import subprocess
import sys
import time
import requests

CONFIG_PATH = "service_config.json"
BASE_URL = "http://localhost:16200/api/v1"
TEST_DATA = """
Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。
它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python拥有一个庞大且 comprehensive 的标准库，涵盖了网络编程、文件处理、数据库接口等众多领域。

近年来，Python在人工智能和机器学习领域获得了巨大的关注。TensorFlow和PyTorch是两个最流行的深度学习框架，
它们都提供了Python接口。NumPy和Pandas是数据科学的基础库，用于数值计算和数据分析。

Django和Flask是两个最受欢迎的Python Web框架。Django提供了"batteries included"的全栈解决方案，
而Flask则是一个轻量级的微框架。FastAPI是近年来崛起的新星，支持异步编程和自动API文档生成。

Python的包管理工具pip使得安装和管理第三方库变得非常简单。虚拟环境工具venv和conda
帮助开发者隔离不同项目的依赖。PyPI是Python的官方第三方软件仓库，拥有超过40万个项目。
"""

RESULTS = []


def set_max_concurrency(n):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    cfg["llm"]["max_concurrency"] = n
    cfg["llm"]["extraction"]["max_concurrency"] = n
    cfg["llm"]["alignment"]["max_concurrency"] = n
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def start_server():
    log = open("/tmp/dd_bench_server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.server.api", "--config", CONFIG_PATH],
        stdout=log, stderr=subprocess.STDOUT,
    )
    # Wait for server ready
    for _ in range(120):
        time.sleep(2)
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.ok:
                return proc
        except Exception:
            pass
    proc.kill()
    raise RuntimeError("Server failed to start")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    time.sleep(2)


def run_remember(graph_id):
    r = requests.post(
        f"{BASE_URL}/remember",
        params={"graph_id": graph_id},
        json={"text": TEST_DATA, "source": "benchmark"},
        timeout=600,
    )
    return r.json()


def wait_for_task(task_id, graph_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(
                f"{BASE_URL}/remember/tasks/{task_id}",
                params={"graph_id": graph_id},
                timeout=10,
            )
            data = r.json()
            status = data.get("data", {}).get("status", "")
            if status == "completed":
                return data["data"]
            if status == "failed":
                return {"status": "failed", "error": data.get("data", {}).get("error")}
        except Exception:
            pass
        time.sleep(2)
    return {"status": "timeout"}


def extract_timings(server_log):
    """Extract step timings from server log."""
    timings = {}
    step_pattern = re.compile(r'\[(step\d+[_\w]*)\].*?(\d+\.\d+)s')
    with open(server_log, "r") as f:
        for line in f:
            for m in step_pattern.finditer(line):
                key = m.group(1)
                val = float(m.group(2))
                timings[key] = val
    return timings


def cleanup_graph(graph_id):
    try:
        requests.post(f"{BASE_URL}/graphs/{graph_id}/clear", params={"graph_id": graph_id}, timeout=10)
    except Exception:
        pass


def benchmark_one(threads):
    print(f"\n{'='*60}")
    print(f"  Benchmark: {threads} thread(s)")
    print(f"{'='*60}")

    set_max_concurrency(threads)
    graph_id = f"bench_t{threads}_{int(time.time())}"

    proc = start_server()
    try:
        # Submit remember task
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/remember",
            params={"graph_id": graph_id},
            json={"text": TEST_DATA, "source": f"benchmark-{threads}t"},
            timeout=30,
        )
        resp = r.json()
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            print(f"  ERROR: no task_id in response: {resp}")
            return None

        # Wait for completion
        result = wait_for_task(task_id, graph_id, timeout=600)
        elapsed = time.time() - t0

        if result.get("status") == "failed":
            print(f"  FAILED: {result.get('error', 'unknown')}")
            return None

        # Extract timings from log
        timings = extract_timings("/tmp/dd_bench_server.log")

        # Get graph stats
        try:
            stats_r = requests.get(f"{BASE_URL}/graph/stats", params={"graph_id": graph_id}, timeout=10)
            stats = stats_r.json().get("data", {})
        except Exception:
            stats = {}

        record = {
            "threads": threads,
            "total_seconds": round(elapsed, 2),
            "entities": stats.get("entity_count", "?"),
            "relations": stats.get("relation_count", "?"),
            "timings": timings,
        }
        RESULTS.append(record)

        # Print summary
        print(f"  Total: {elapsed:.1f}s | Entities: {record['entities']} | Relations: {record['relations']}")
        if timings:
            top = sorted(timings.items(), key=lambda x: -x[1])[:8]
            for k, v in top:
                print(f"    {k}: {v:.2f}s")

        cleanup_graph(graph_id)
        return record
    finally:
        stop_server(proc)


def main():
    thread_counts = [1, 4, 8, 16, 32]

    print("=== Neo4j Connection Optimization Benchmark ===")
    print(f"LLM: xinference (port 9997), model: gemma4-26b-32k")
    print(f"Thread counts to test: {thread_counts}")
    print()

    for t in thread_counts:
        benchmark_one(t)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threads':>8} | {'Total(s)':>10} | {'Entities':>8} | {'Relations':>9}")
    print("-" * 50)
    for r in RESULTS:
        print(f"{r['threads']:>8} | {r['total_seconds']:>10.1f} | {r['entities']:>8} | {r['relations']:>9}")

    # Speedup
    if len(RESULTS) >= 2:
        base = RESULTS[0]["total_seconds"]
        print(f"\nSpeedup vs 1-thread:")
        for r in RESULTS:
            speedup = base / r["total_seconds"] if r["total_seconds"] > 0 else 0
            print(f"  {r['threads']:>2}t: {speedup:.2f}x")

    # Save results
    with open("benchmark_results_threads.json", "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to benchmark_results_threads.json")


if __name__ == "__main__":
    main()
