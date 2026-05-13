#!/usr/bin/env python3
"""Benchmark: 1/4/8/16/32 threads — each run writes to bench_results/{N}t/"""

import json, os, re, subprocess, sys, time, requests, shutil

CONFIG_PATH = "service_config.json"
BASE_URL = "http://localhost:16200/api/v1"
RESULTS_DIR = "bench_results"
SERVER_LOG = "/tmp/dd_bench_server.log"

TEST_TEXT = """
Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。
它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python拥有一个庞大且comprehensive的标准库，涵盖了网络编程、文件处理、数据库接口等众多领域。

近年来，Python在人工智能和机器学习领域获得了巨大的关注。TensorFlow和PyTorch是两个最流行的深度学习框架，
它们都提供了Python接口。NumPy和Pandas是数据科学的基础库，用于数值计算和数据分析。

Django和Flask是两个最受欢迎的Python Web框架。Django提供了"batteries included"的全栈解决方案，
而Flask则是一个轻量级的微框架。FastAPI是近年来崛起的新星，支持异步编程和自动API文档生成。

Python的包管理工具pip使得安装和管理第三方库变得非常简单。虚拟环境工具venv和conda
帮助开发者隔离不同项目的依赖。PyPI是Python的官方第三方软件仓库，拥有超过40万个项目。
"""

THREAD_COUNTS = [1, 4, 8, 16, 32]


def set_max_concurrency(n):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["llm"]["max_concurrency"] = n
    cfg["llm"]["extraction"]["max_concurrency"] = n
    cfg["llm"]["alignment"]["max_concurrency"] = n
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def start_server(log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.server.api", "--config", CONFIG_PATH],
        stdout=log, stderr=subprocess.STDOUT,
    )
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


def wait_for_task(task_id, graph_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(
                f"{BASE_URL}/remember/tasks/{task_id}",
                params={"graph_id": graph_id}, timeout=10,
            )
            data = r.json().get("data", {})
            status = data.get("status", "")
            if status == "completed":
                return data
            if status in ("failed", "error"):
                return {"status": "failed", "error": data.get("error")}
        except Exception:
            pass
        time.sleep(2)
    return {"status": "timeout"}


def parse_step_timings(log_path):
    """Extract all step timings from server log."""
    timings = {}
    # Match patterns like 【步骤X】...｜XXs｜  or [stepN_timing] xxx: XX.XXs
    step_re = re.compile(r'【步骤(\d+)】[^｜]*｜(\d+\.?\d*)s')
    step_label_re = re.compile(r'【步骤(\d+)】(\w+)[^｜]*｜(\d+\.?\d*)s')
    timing_re = re.compile(r'\[(step\d+\w*)\][^:]*:\s*(\d+\.?\d*)s')

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # 【步骤9】process_entities｜28.7s｜51个实体
            for m in step_label_re.finditer(line):
                step_n = int(m.group(1))
                label = m.group(2)
                val = float(m.group(3))
                key = f"step{step_n}_{label}"
                timings[key] = val
            # 【步骤X】完成｜XXs
            for m in step_re.finditer(line):
                step_n = int(m.group(1))
                val = float(m.group(2))
                timings[f"step{step_n}_total"] = val
            # [step10b-process_loop]: 5.23s
            for m in timing_re.finditer(line):
                timings[m.group(1)] = float(m.group(2))

    return timings


def parse_entity_relation_counts(log_path):
    """Extract entity/relation counts from server log."""
    counts = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # 【步骤9】小结｜实体｜唯一51·原51
            m = re.search(r'实体｜唯一(\d+)', line)
            if m:
                counts["entities_unique"] = int(m.group(1))
            # 【步骤10】关系｜待处理｜去重(\d+)
            m = re.search(r'去重(\d+)', line)
            if m:
                counts["relations_unique"] = int(m.group(1))
            # 增量保存 X 条关系
            m = re.search(r'增量保存\s*(\d+)\s*条关系', line)
            if m:
                counts["relations_saved"] = int(m.group(1))
            # saved incrementally
            m = re.search(r'(\d+)\s*saved incrementally', line)
            if m:
                counts["relations_saved_inc"] = int(m.group(1))
    return counts


def parse_errors(log_path):
    """Check for Neo4j connection errors."""
    errors = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "ServiceUnavailable" in line or "Couldn't connect" in line:
                errors.append(line.strip()[:200])
    return errors


def benchmark_one(threads):
    run_dir = f"{RESULTS_DIR}/{threads}t"
    os.makedirs(run_dir, exist_ok=True)
    log_path = f"{run_dir}/server.log"

    print(f"\n{'='*60}")
    print(f"  Benchmark: {threads} thread(s)")
    print(f"{'='*60}")
    sys.stdout.flush()

    set_max_concurrency(threads)
    graph_id = f"bench_t{threads}_{int(time.time())}"

    proc = start_server(log_path)
    try:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/remember",
            params={"graph_id": graph_id},
            json={"text": TEST_TEXT, "source": f"bench-{threads}t"},
            timeout=30,
        )
        resp = r.json()
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            print(f"  ERROR: {resp}")
            return None

        result = wait_for_task(task_id, graph_id, timeout=600)
        elapsed = time.time() - t0

        # Parse log
        timings = parse_step_timings(log_path)
        counts = parse_entity_relation_counts(log_path)
        errors = parse_errors(log_path)

        record = {
            "threads": threads,
            "graph_id": graph_id,
            "total_seconds": round(elapsed, 2),
            "counts": counts,
            "timings": timings,
            "neo4j_errors": errors[:10],
        }

        # Save individual result
        with open(f"{run_dir}/result.json", "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        # Save readable summary
        with open(f"{run_dir}/summary.txt", "w") as f:
            f.write(f"Threads: {threads}\n")
            f.write(f"Total time: {elapsed:.1f}s\n")
            f.write(f"Entities: {counts.get('entities_unique', '?')}\n")
            f.write(f"Relations: {counts.get('relations_unique', '?')}\n")
            f.write(f"Neo4j errors: {len(errors)}\n\n")
            f.write("Step timings:\n")
            for k, v in sorted(timings.items()):
                f.write(f"  {k}: {v:.2f}s\n")
            if errors:
                f.write(f"\nNeo4j connection errors ({len(errors)}):\n")
                for e in errors[:5]:
                    f.write(f"  {e}\n")

        # Cleanup graph
        try:
            requests.post(f"{BASE_URL}/graphs/{graph_id}/clear",
                          params={"graph_id": graph_id}, timeout=10)
        except Exception:
            pass

        print(f"  Total: {elapsed:.1f}s | Entities: {counts.get('entities_unique', '?')} | "
              f"Relations: {counts.get('relations_unique', '?')} | Neo4j errors: {len(errors)}")
        sys.stdout.flush()
        return record
    finally:
        stop_server(proc)


def print_final_report(all_results):
    report_path = f"{RESULTS_DIR}/report.txt"
    lines = []

    lines.append("=" * 70)
    lines.append("  BENCHMARK REPORT — Neo4j Connection Optimization")
    lines.append("  LLM: xinference (port 9997), model: gemma4-26b-32k")
    lines.append(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append(f"{'Threads':>8} | {'Total(s)':>10} | {'Entities':>8} | {'Relations':>9} | {'Errors':>6}")
    lines.append("-" * 60)
    for r in all_results:
        c = r.get("counts", {})
        lines.append(
            f"{r['threads']:>8} | {r['total_seconds']:>10.1f} | "
            f"{c.get('entities_unique', '?'):>8} | {c.get('relations_unique', '?'):>9} | "
            f"{len(r.get('neo4j_errors', [])):>6}"
        )

    # Speedup
    if all_results:
        base = all_results[0]["total_seconds"]
        lines.append("")
        lines.append("Speedup vs 1-thread:")
        for r in all_results:
            sp = base / r["total_seconds"] if r["total_seconds"] > 0 else 0
            lines.append(f"  {r['threads']:>2}t: {sp:.2f}x ({r['total_seconds']:.1f}s)")

    # Per-step comparison
    lines.append("")
    lines.append("=" * 70)
    lines.append("  STEP-BY-STEP COMPARISON")
    lines.append("=" * 70)

    # Collect all step keys
    all_keys = set()
    for r in all_results:
        all_keys.update(r.get("timings", {}).keys())
    all_keys = sorted(all_keys)

    if all_keys:
        header = f"{'Step':<35}"
        for r in all_results:
            header += f" | {r['threads']:>2}t"
        lines.append(header)
        lines.append("-" * len(header))

        for key in all_keys:
            row = f"{key:<35}"
            for r in all_results:
                val = r.get("timings", {}).get(key, 0)
                row += f" | {val:>5.1f}s"
            lines.append(row)

    # Error summary
    has_errors = any(r.get("neo4j_errors") for r in all_results)
    if has_errors:
        lines.append("")
        lines.append("=" * 70)
        lines.append("  NEO4J CONNECTION ERRORS")
        lines.append("=" * 70)
        for r in all_results:
            errs = r.get("neo4j_errors", [])
            if errs:
                lines.append(f"\n  {r['threads']}t: {len(errs)} errors")
                for e in errs[:3]:
                    lines.append(f"    {e}")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    return report


def main():
    shutil.rmtree(RESULTS_DIR, ignore_errors=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Multi-Thread Benchmark ===")
    print(f"Results dir: {RESULTS_DIR}/")
    print(f"Thread counts: {THREAD_COUNTS}")
    sys.stdout.flush()

    all_results = []
    for t in THREAD_COUNTS:
        r = benchmark_one(t)
        if r:
            all_results.append(r)

    report = print_final_report(all_results)

    # Save JSON
    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
