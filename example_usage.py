"""
使用示例：通过 HTTP API 操作 Temporal Memory Graph

前置条件：
  1. 启动 API 服务：python service_api.py --config service_config.json

本脚本演示两个核心场景：
  1. Remember — 批量传文本记忆（含 event_time）
  2. Find — 语义检索唤醒局部记忆
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "http://127.0.0.1:16200"


def pp(label: str, resp: requests.Response):
    """格式化打印 API 响应"""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {resp.request.method} {resp.request.url}")
    print(f"  Status: {resp.status_code}")
    print(f"{'=' * 60}")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text[:500])
    print()


def _fmt_ts(ts):
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _fmt_sec(value):
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _print_task_timing(label: str, task_data: dict, wall_elapsed: float | None = None):
    """打印 remember 任务的真实耗时拆解。"""
    created_at = task_data.get("created_at")
    started_at = task_data.get("started_at")
    finished_at = task_data.get("finished_at")
    queue_wait = None
    process_elapsed = None
    total_elapsed = None
    if created_at and started_at:
        queue_wait = started_at - created_at
    if started_at and finished_at:
        process_elapsed = finished_at - started_at
    if created_at and finished_at:
        total_elapsed = finished_at - created_at

    print(f"\n>>> {label} 时间统计")
    print(f"  created_at : {_fmt_ts(created_at)}")
    print(f"  started_at : {_fmt_ts(started_at)}")
    print(f"  finished_at: {_fmt_ts(finished_at)}")
    print(f"  排队耗时   : {_fmt_sec(queue_wait)}")
    print(f"  处理耗时   : {_fmt_sec(process_elapsed)}")
    print(f"  服务总耗时 : {_fmt_sec(total_elapsed)}")
    if wall_elapsed is not None:
        print(f"  客户端总耗时: {_fmt_sec(wall_elapsed)}")


def _print_queue_snapshot(limit: int = 5):
    """打印队列快照，便于观察 remember 并发/排队情况。"""
    resp = requests.get(f"{API_BASE}/api/remember/queue", params={"limit": limit})
    pp("Remember Queue", resp)
    return resp.json()


# ------------------------------------------------------------------
# 0. 健康检查
# ------------------------------------------------------------------
def check_health():
    print("\n>>> 健康检查")
    resp = requests.get(f"{API_BASE}/health")
    pp("Health", resp)
    if resp.status_code != 200 or not resp.json().get("success"):
        print("服务不可用，请先启动 service_api.py")
        sys.exit(1)


def check_llm():
    """检查大模型是否可访问；若服务端无 /health/llm（404）则跳过预检并继续执行。"""
    print("\n>>> 大模型可用性检查")
    try:
        resp = requests.get(f"{API_BASE}/health/llm", timeout=20)
    except Exception as e:
        print(f"请求 /health/llm 失败: {e}")
        print("跳过 LLM 预检，继续执行（若后续 remember 失败请检查服务端 LLM 配置与网络）。")
        return
    pp("LLM Health", resp)
    if resp.status_code == 404:
        print("当前服务未提供 /health/llm 接口（404），可能为旧版或其它入口启动。")
        print("跳过 LLM 预检，继续执行（若后续 remember 失败请用最新 service_api.py 启动并检查 LLM 配置）。")
        return
    if resp.status_code != 200:
        # 优先从 JSON 取服务端返回的 error/message
        try:
            data = resp.json()
            server_msg = data.get("error") or data.get("message") or ""
        except Exception:
            server_msg = ""
        # 根据状态码给出明确原因
        if resp.status_code == 503:
            reason = server_msg or "服务端返回 503，大模型或网络不可用。"
        else:
            reason = server_msg or f"HTTP {resp.status_code}"
        print("大模型不可用原因:", reason)
        if server_msg and server_msg != reason:
            print("服务端报错:", server_msg)
        elif not server_msg and resp.text and len(resp.text) < 500:
            print("原始响应:", resp.text.strip()[:300])
        print("建议: 检查 service_config.json 中的 llm.api_key / llm.base_url / llm.model，确认网络与配置无误。")
        sys.exit(1)
    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if not data.get("data", {}).get("llm_available", True):
        print("大模型不可用原因: 大模型未就绪或未返回有效结果，跳过后续示例。")
        sys.exit(1)
    print("大模型访问正常，继续执行示例。")


# ------------------------------------------------------------------
# 1. Remember — 批量传文本（推荐方式）
# ------------------------------------------------------------------
def _poll_task(task_id: str, label: str, timeout: int = 600):
    """轮询异步 remember 任务直到完成或超时。"""
    wall_start = time.time()
    deadline = time.time() + timeout
    interval = 1
    poll_count = 0
    while time.time() < deadline:
        resp = requests.get(f"{API_BASE}/api/remember/status/{task_id}")
        data = resp.json().get("data", {})
        status = data.get("status", "unknown")
        poll_count += 1
        created_at = data.get("created_at")
        started_at = data.get("started_at")
        finished_at = data.get("finished_at")
        queue_wait = (started_at - created_at) if (created_at and started_at) else None
        process_elapsed = (finished_at - started_at) if (started_at and finished_at) else None
        if status in ("completed", "failed"):
            pp(f"{label} (最终状态)", resp)
            _print_task_timing(label, data, wall_elapsed=time.time() - wall_start)
            return data
        print(
            f"  … {label} 状态[{poll_count}]: {status}"
            f", 排队={_fmt_sec(queue_wait)}"
            f", 处理中={_fmt_sec(process_elapsed)}"
            f", 下一次轮询等待 {interval}s"
        )
        time.sleep(interval)
        interval = min(interval * 2, 10)
    print(f"  ⚠ {label} 超时，客户端已等待 {_fmt_sec(time.time() - wall_start)}")
    return None


def example_remember_text():
    print("\n>>> Remember: 批量传文本记忆（异步）")

    text = (
        "罗辑是一名社会学教授，他被选为面壁者之一。"
        "面壁计划是人类为了对抗三体入侵而制定的战略防御计划，"
        "允许面壁者在不向任何人解释的情况下调动大量资源。"
        "罗辑最初对自己被选为面壁者感到困惑，因为他既不是科学家也不是军事家。"
        "后来他意识到，叶文洁曾经告诉他宇宙社会学的两条公理和两个重要概念，"
        "这可能是他被选中的真正原因。"
    )

    submit_start = time.time()
    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": text,
            "source_name": "三体测试-文本",
            "event_time": "2026-03-09T14:00:00",
        },
    )
    submit_elapsed = time.time() - submit_start
    pp("Remember Text (提交)", resp)
    print(f"  提交请求耗时: {_fmt_sec(submit_elapsed)}")
    data = resp.json().get("data", {})
    task_id = data.get("task_id")
    if task_id:
        _print_queue_snapshot()
        _poll_task(task_id, "Remember Text")
    return resp.json()


# ------------------------------------------------------------------
# 2. Remember — 传长文本（模拟读完一篇文章后整段写入）
# ------------------------------------------------------------------
def example_remember_long():
    print("\n>>> Remember: 传长文本（异步）")

    long_text = """今天下午14点到16点，我阅读了《三体2：黑暗森林》的前三章。

第一章"面壁者"：联合国特别会议上宣布了面壁计划，四位面壁者被选出。罗辑作为一个普通的社会学教授，意外地成为了面壁者之一。其他三位面壁者分别是前美国国防部长泰勒、委内瑞拉总统雷迪亚兹和英国科学家比尔·希恩斯。

第二章"咒语"：罗辑回忆起多年前与叶文洁的一次深夜谈话。叶文洁向他提到了宇宙社会学的两条基本公理：一、生存是文明的第一需要；二、文明不断增长和扩张，但宇宙中的物质总量保持不变。她还提出了两个重要概念：猜疑链和技术爆炸。

第三章"破壁人"：三体世界为每位面壁者指派了一个破壁人，负责分析和破解面壁者的真实计划。罗辑的破壁人是一个看起来很普通的年轻人。"""

    submit_start = time.time()
    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": long_text,
            "source_name": "阅读日志-三体2",
            "event_time": "2026-03-09T16:00:00",
        },
    )
    submit_elapsed = time.time() - submit_start
    pp("Remember Long Text (提交)", resp)
    print(f"  提交请求耗时: {_fmt_sec(submit_elapsed)}")
    data = resp.json().get("data", {})
    task_id = data.get("task_id")
    if task_id:
        _print_queue_snapshot()
        _poll_task(task_id, "Remember Long Text")
    return resp.json()


# ------------------------------------------------------------------
# 2c. Remember — 超长文本：直接读取《三体2黑暗森林》全文
# ------------------------------------------------------------------
# 默认路径：项目根下 datas/docs/三体2黑暗森林.txt（可被环境变量 OVERRIDE_ULTRALONG_PATH 覆盖）
_ULTRALONG_FILE = Path(__file__).resolve().parent.parent / "datas" / "docs" / "三体2黑暗森林.txt"


def example_remember_ultralong():
    """读取本地《三体2黑暗森林》全文，提交 remember，入队即返回后轮询直到完成。"""
    print("\n>>> Remember: 超长文本（三体2黑暗森林.txt 全文）")

    file_path = Path(os.environ.get("OVERRIDE_ULTRALONG_PATH", str(_ULTRALONG_FILE)))
    if not file_path.exists():
        print(f"  跳过：文件不存在 {file_path}")
        print("  可设置环境变量 OVERRIDE_ULTRALONG_PATH 指定其它路径。")
        return

    text = file_path.read_text(encoding="utf-8")
    char_count = len(text)
    print(f"  文件: {file_path}")
    print(f"  字符数: {char_count}")

    submit_start = time.time()
    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": text,
            "source_name": "三体2黑暗森林-全文",
            "event_time": datetime.now().isoformat(),
        },
    )
    submit_elapsed = time.time() - submit_start
    pp("Remember 超长文本 (提交)", resp)
    print(f"  提交请求耗时: {_fmt_sec(submit_elapsed)}")
    data = resp.json().get("data", {})
    task_id = data.get("task_id")
    if task_id:
        _print_queue_snapshot()
        _poll_task(task_id, "Remember 超长文本", timeout=3600)
    return resp.json()


# ------------------------------------------------------------------
# 2b. Remember — 并发测试：第一个在处理时第二个到来
# ------------------------------------------------------------------
def example_remember_concurrent():
    """先提交长任务 A，间隔几秒后提交短任务 B，轮询两者直到都完成，验证排队与并行行为。
    时间轴说明：存入记忆库的时间以请求中的 event_time 为准，与任务完成先后无关——
    即使 B 先完成、A 后完成，实体/关系版本顺序仍按 event_time（A 16:00，B 16:01）正确排列。"""
    print("\n>>> Remember 并发测试：第一个在处理时第二个到来")

    long_text = """今天下午14点到16点，我阅读了《三体2：黑暗森林》的前三章。
第一章"面壁者"：联合国特别会议上宣布了面壁计划，四位面壁者被选出。罗辑作为一个普通的社会学教授，意外地成为了面壁者之一。
第二章"咒语"：罗辑回忆起多年前与叶文洁的一次深夜谈话。叶文洁向他提到了宇宙社会学的两条基本公理。
第三章"破壁人"：三体世界为每位面壁者指派了一个破壁人，负责分析和破解面壁者的真实计划。"""
    short_text = "罗辑是面壁者之一。叶文洁向他提到过宇宙社会学公理。"

    # 1) 提交任务 A（长文本）
    t0 = time.time()
    resp_a = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": long_text,
            "source_name": "并发测试-长",
            "event_time": "2026-03-09T16:00:00",
        },
    )
    submit_a = time.time() - t0
    pp("任务 A (长文本) 提交", resp_a)
    print(f"  任务 A 提交耗时: {_fmt_sec(submit_a)}")
    data_a = resp_a.json().get("data", {})
    task_id_a = data_a.get("task_id")
    if not task_id_a:
        print("任务 A 未返回 task_id，跳过并发测试。")
        return

    _print_queue_snapshot()

    # 2) 间隔几秒，让 A 进入处理中，再提交任务 B（短文本）
    delay_b = 30
    print(f"\n  等待 {delay_b}s 后提交任务 B（模拟：第一个在处理时第二个到来）…")
    time.sleep(delay_b)

    t1 = time.time()
    resp_b = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": short_text,
            "source_name": "并发测试-短",
            "event_time": "2026-03-09T16:01:00",
        },
    )
    submit_b = time.time() - t1
    pp("任务 B (短文本) 提交", resp_b)
    print(f"  任务 B 提交耗时: {_fmt_sec(submit_b)}")
    data_b = resp_b.json().get("data", {})
    task_id_b = data_b.get("task_id")
    if not task_id_b:
        print("任务 B 未返回 task_id，仅轮询任务 A。")

    _print_queue_snapshot()

    # 3) 同时轮询 A、B 直到都结束
    def get_status(tid):
        r = requests.get(f"{API_BASE}/api/remember/status/{tid}")
        return r.json().get("data", {})

    deadline = time.time() + 600
    interval = 1
    poll_count = 0
    done_a, done_b = False, not task_id_b  # 无 B 时视为 B 已“完成”
    results = {"a": None, "b": None}

    while time.time() < deadline:
        poll_count += 1
        if not done_a:
            data_a = get_status(task_id_a)
            sa = data_a.get("status", "unknown")
            done_a = sa in ("completed", "failed")
            results["a"] = data_a
        if task_id_b and not done_b:
            data_b = get_status(task_id_b)
            sb = data_b.get("status", "unknown")
            done_b = sb in ("completed", "failed")
            results["b"] = data_b

        print(
            f"  … 轮询[{poll_count}] "
            f"A={results['a'].get('status', '?')} "
            + (f" B={results['b'].get('status', '?')}" if task_id_b else "")
        )
        if done_a and done_b:
            break
        time.sleep(interval)
        interval = min(interval * 2, 10)

    # 4) 打印两个任务的最终状态与时间统计
    print("\n" + "=" * 60)
    print("  并发测试结果汇总")
    print("=" * 60)
    _print_task_timing("任务 A (长)", results["a"])
    if results["b"]:
        _print_task_timing("任务 B (短)", results["b"])
    _print_queue_snapshot()

    # 简单断言：两个都应完成
    if results["a"] and results["a"].get("status") != "completed":
        print("  ⚠ 任务 A 未 completed:", results["a"].get("status"))
    if results["b"] and results["b"].get("status") != "completed":
        print("  ⚠ 任务 B 未 completed:", results["b"].get("status"))
    if (results["a"] or {}).get("status") == "completed" and (not results["b"] or results["b"].get("status") == "completed"):
        print("  ✓ 两个任务均按预期完成（或仅 A 完成）。")
    # 时间轴按 event_time：A=16:00、B=16:01，与完成先后无关；实体/关系版本顺序以 physical_time 为准。


# ------------------------------------------------------------------
# 3. Find — 统一语义检索
# ------------------------------------------------------------------
def example_find():
    print("\n>>> Find: 语义检索")

    queries = [
        "罗辑为什么被选为面壁者",
        "面壁计划是什么",
        "叶文洁和罗辑的关系",
    ]

    for q in queries:
        resp = requests.post(
            f"{API_BASE}/api/find",
            json={
                "query": q,
                "max_entities": 10,
                "max_relations": 20,
                "expand": True,
            },
        )
        pp(f"Find: {q}", resp)
        time.sleep(0.5)


# ------------------------------------------------------------------
# 4. Find — 原子接口示例
# ------------------------------------------------------------------
def example_find_atomic():
    print("\n>>> Find: 原子接口")

    resp = requests.get(f"{API_BASE}/api/find/stats")
    pp("Stats", resp)

    resp = requests.get(
        f"{API_BASE}/api/find/entities/search",
        params={"query_name": "罗辑", "max_results": 5, "threshold": 0.3},
    )
    pp("Entity Search: 罗辑", resp)

    resp = requests.get(
        f"{API_BASE}/api/find/relations/search",
        params={"query_text": "面壁者", "max_results": 5, "threshold": 0.3},
    )
    pp("Relation Search: 面壁者", resp)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Temporal Memory Graph — API 使用示例")
    print("=" * 60)

    check_health()
    check_llm()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("text", "all"):
        example_remember_text()

    if mode in ("long", "all"):
        example_remember_long()

    if mode == "ultralong":
        example_remember_ultralong()

    if mode == "concurrent":
        example_remember_concurrent()

    if mode in ("find", "all"):
        example_find()

    if mode in ("atomic", "all"):
        example_find_atomic()

    if mode not in ("text", "long", "ultralong", "concurrent", "find", "atomic", "all"):
        print(f"""
用法: python example_usage.py [mode]

mode 可选值:
  text       — 测试文本记忆（含 event_time）
  long       — 测试长文本记忆（模拟阅读日志）
  ultralong  — 超长文本：读取 datas/docs/三体2黑暗森林.txt 全文并 remember
  concurrent — 并发测试：第一个 remember 在处理时第二个到来（排队/并行）
  find       — 测试语义检索
  atomic     — 测试原子接口
  all        — 运行全部示例（默认）
""")
