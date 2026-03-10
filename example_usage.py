"""
使用示例：通过 HTTP API 操作 Temporal Memory Graph

前置条件：
  1. 启动 API 服务：python service_api.py --config service_config.json

本脚本演示两个核心场景：
  1. Remember — 批量传文本记忆（含 event_time）
  2. Find — 语义检索唤醒局部记忆
"""
import json
import sys
import time

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
    """检查大模型是否可访问，不可用则退出，不执行后续示例。"""
    print("\n>>> 大模型可用性检查")
    resp = requests.get(f"{API_BASE}/health/llm", timeout=20)
    pp("LLM Health", resp)
    if resp.status_code != 200:
        # 优先从 JSON 取服务端返回的 error/message
        try:
            data = resp.json()
            server_msg = data.get("error") or data.get("message") or ""
        except Exception:
            server_msg = ""
        # 根据状态码给出明确原因
        if resp.status_code == 404:
            reason = "接口不存在 (404)。请重启 service_api.py 以加载 /health/llm 路由后再运行示例。"
        elif resp.status_code == 503:
            reason = server_msg or "服务端返回 503，大模型或网络不可用。"
        else:
            reason = server_msg or f"HTTP {resp.status_code}"
        # 报错信息：原因 + 若有原始内容则简要给出
        print("大模型不可用原因:", reason)
        if server_msg and server_msg != reason:
            print("服务端报错:", server_msg)
        elif not server_msg and resp.text and len(resp.text) < 500:
            print("原始响应:", resp.text.strip()[:300])
        print("建议: 检查 service_config.json 中的 llm.api_key / llm.base_url / llm.model，确认网络与配置无误；若为 404 请重启 API 服务。")
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
    deadline = time.time() + timeout
    interval = 1
    while time.time() < deadline:
        resp = requests.get(f"{API_BASE}/api/remember/status/{task_id}")
        data = resp.json().get("data", {})
        status = data.get("status", "unknown")
        if status in ("completed", "failed"):
            pp(f"{label} (最终状态)", resp)
            return data
        print(f"  … {label} 状态: {status}，等待 {interval}s")
        time.sleep(interval)
        interval = min(interval * 2, 10)
    print(f"  ⚠ {label} 超时")
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

    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": text,
            "source_name": "三体测试-文本",
            "event_time": "2026-03-09T14:00:00",
        },
    )
    pp("Remember Text (提交)", resp)
    data = resp.json().get("data", {})
    task_id = data.get("task_id")
    if task_id:
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

    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={
            "text": long_text,
            "source_name": "阅读日志-三体2",
            "event_time": "2026-03-09T16:00:00",
        },
    )
    pp("Remember Long Text (提交)", resp)
    data = resp.json().get("data", {})
    task_id = data.get("task_id")
    if task_id:
        _poll_task(task_id, "Remember Long Text")
    return resp.json()


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

    if mode in ("find", "all"):
        example_find()

    if mode in ("atomic", "all"):
        example_find_atomic()

    if mode not in ("text", "long", "find", "atomic", "all"):
        print(f"""
用法: python example_usage.py [mode]

mode 可选值:
  text    — 测试文本记忆（含 event_time）
  long    — 测试长文本记忆（模拟阅读日志）
  find    — 测试语义检索
  atomic  — 测试原子接口
  all     — 运行全部示例（默认）
""")
