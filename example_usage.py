"""
使用示例：通过 HTTP API 操作 Temporal Memory Graph

前置条件：
  1. 启动 API 服务：python service_api.py --config service_config.json

本脚本演示两个核心场景：
  1. Remember — 通过 GET /api/remember 批量传文本记忆（含 event_time）
  2. Find — 语义检索唤醒局部记忆
"""
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

API_BASE = "http://127.0.0.1:16200"

# 超过该长度改用 text_b64，减轻 URL 编码体积与部分代理限制
_REMEMBER_TEXT_B64_THRESHOLD = 4000


def _remember_get(
    text: str,
    source_name: str = "api_input",
    event_time: Optional[str] = None,
    load_cache_memory: Optional[bool] = None,
) -> requests.Response:
    """调用 GET /api/remember（仅 GET，不再使用 POST）。"""
    params: dict = {"source_name": source_name}
    if event_time:
        params["event_time"] = event_time
    if load_cache_memory is not None:
        params["load_cache_memory"] = "true" if load_cache_memory else "false"
    if len(text) > _REMEMBER_TEXT_B64_THRESHOLD:
        params["text_b64"] = base64.b64encode(text.encode("utf-8")).decode("ascii")
    else:
        params["text"] = text
    return requests.get(f"{API_BASE}/api/remember", params=params, timeout=300)


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
    resp = _remember_get(
        text,
        source_name="三体测试-文本",
        event_time="2026-03-09T14:00:00",
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
    resp = _remember_get(
        long_text,
        source_name="阅读日志-三体2",
        event_time="2026-03-09T16:00:00",
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
    resp = _remember_get(
        text,
        source_name="三体2黑暗森林-全文",
        event_time=datetime.now().isoformat(),
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
# 2b. Remember — 并发测试：列表驱动，隔几秒提交一个，再轮询全部
# ------------------------------------------------------------------
# 并发测试数据：每项提交一次 remember，项数可变；相邻两次提交间隔 delay_between_submit_seconds 秒
CONCURRENT_TASKS = [
    {
        "label": "任务1-长",
        "text": """今天下午14点到16点，我阅读了《三体2：黑暗森林》的前三章。
第一章"面壁者"：联合国特别会议上宣布了面壁计划，四位面壁者被选出。罗辑作为一个普通的社会学教授，意外地成为了面壁者之一。
第二章"咒语"：罗辑回忆起多年前与叶文洁的一次深夜谈话。叶文洁向他提到了宇宙社会学的两条基本公理。
第三章"破壁人"：三体世界为每位面壁者指派了一个破壁人，负责分析和破解面壁者的真实计划。""",
        "source_name": "并发测试-长",
        "event_time": "2026-03-09T16:00:00",
    },
    {
        "label": "任务2-短",
        "text": """# TMG Memory Graph — Agent 长期记忆技能

> TMG 是一个为 Agent 设计的时序自然语言记忆图。
> 你不是在调用一个工具——你是在使用自己的长期记忆。

## 技能目录结构（按需加载）

```
skills/tmg-memory-graph/
├── SKILL.md           # 本文件：核心指令 + 元数据
├── reference.md       # 接口路径、请求/响应、原子端点
├── templates/         # 格式模板（按需读取）
│   └── remember-format.md
└── examples/          # 优秀示例（按需读取）
    ├── work-session.md
    ├── read-document.md
    ├── daily-reflection.md
    └── before-reading.md
```

需要具体格式或示例时再读对应文件，避免一次性塞满上下文。

## 使用场景

在以下情况触发此技能：

```""",
        "source_name": "并发测试-短",
        "event_time": "2026-03-09T16:01:00",
    },
    {
        "label": "任务3-长",
        "text": """# TMG Memory Graph — Agent 长期记忆技能

> TMG 是一个为 Agent 设计的时序自然语言记忆图。
> 你不是在调用一个工具——你是在使用自己的长期记忆。

## 技能目录结构（按需加载）

```
skills/tmg-memory-graph/
├── SKILL.md           # 本文件：核心指令 + 元数据
├── reference.md       # 接口路径、请求/响应、原子端点
├── templates/         # 格式模板（按需读取）
│   └── remember-format.md
└── examples/          # 优秀示例（按需读取）
    ├── work-session.md
    ├── read-document.md
    ├── daily-reflection.md
    └── before-reading.md
```

需要具体格式或示例时再读对应文件，避免一次性塞满上下文。

## 使用场景

在以下情况触发此技能：

| 场景 | 示例 |
|------|------|
| 记忆写入 | 「把今天做的事情记下来」「读完这篇论文后存入记忆」 |
| 记忆检索 | 「之前关于 XX 的讨论内容是什么」「上周我在做什么」 |
| 服务部署 | 「启动 TMG」「帮我配置记忆服务」 |
| 身份集成 | 「把记忆能力写进 SOUL.md」「配置心跳记忆同步」 |

## 系统简介

- TMG 是**统一的自然语言记忆图**，不是多库/多标签系统
- 所有记忆写入同一张图，系统自动完成概念抽取、关系构建、语义对齐
- 系统只负责 **Remember**（写入）和 **Find**（检索）
- **Select**（筛选与决策）由调用方（你）完成
- 每条记忆带时间戳，实体/关系有版本链，支持时间回溯

## 项目信息

| 项 | 值 |
|----|-----|
| 仓库 | `https://github.com/ngyygm/Temporal_Memory_Graph` |
| 项目根目录 | `Temporal_Memory_Graph/` |
| 依赖文件 | `requirements.txt` |
| 配置文件 | `service_config.json`（模板：`service_config.example.json`） |
| 默认地址 | `http://127.0.0.1:16200` |
| 健康检查 | `GET /health` |
| 启动命令 | `python service_api.py --config service_config.json` |

## 部署与启动

按顺序执行，已完成的步骤可跳过。

### 1. 克隆仓库（如本地不存在）

```bash
git clone https://github.com/ngyygm/Temporal_Memory_Graph
cd Temporal_Memory_Graph
```""",
        "source_name": "并发测试-短",
        "event_time": "2026-03-09T16:01:00",
    },
    {
        "label": "任务4-长",
        "text": """今天下午14点到16点，我阅读了《三体2：黑暗森林》的前三章。
第一章"面壁者"：联合国特别会议上宣布了面壁计划，四位面壁者被选出。罗辑作为一个普通的社会学教授，意外地成为了面壁者之一。
第二章"咒语"：罗辑回忆起多年前与叶文洁的一次深夜谈话。叶文洁向他提到了宇宙社会学的两条基本公理。
第三章"破壁人"：三体世界为每位面壁者指派了一个破壁人，负责分析和破解面壁者的真实计划。""",
        "source_name": "并发测试-长",
        "event_time": "2026-03-09T16:00:00",
    },
    {
        "label": "任务5-短",
        "text": """# TMG Memory Graph — Agent 长期记忆技能

> TMG 是一个为 Agent 设计的时序自然语言记忆图。
> 你不是在调用一个工具——你是在使用自己的长期记忆。

## 技能目录结构（按需加载）

```
skills/tmg-memory-graph/
├── SKILL.md           # 本文件：核心指令 + 元数据
├── reference.md       # 接口路径、请求/响应、原子端点
├── templates/         # 格式模板（按需读取）
│   └── remember-format.md
└── examples/          # 优秀示例（按需读取）
    ├── work-session.md
    ├── read-document.md
    ├── daily-reflection.md
    └── before-reading.md
```

需要具体格式或示例时再读对应文件，避免一次性塞满上下文。

## 使用场景

在以下情况触发此技能：

```""",
        "source_name": "并发测试-短",
        "event_time": "2026-03-09T16:01:00",
    },
    {
        "label": "任务6-长",
        "text": """# TMG Memory Graph — Agent 长期记忆技能

> TMG 是一个为 Agent 设计的时序自然语言记忆图。
> 你不是在调用一个工具——你是在使用自己的长期记忆。

## 技能目录结构（按需加载）

```
skills/tmg-memory-graph/
├── SKILL.md           # 本文件：核心指令 + 元数据
├── reference.md       # 接口路径、请求/响应、原子端点
├── templates/         # 格式模板（按需读取）
│   └── remember-format.md
└── examples/          # 优秀示例（按需读取）
    ├── work-session.md
    ├── read-document.md
    ├── daily-reflection.md
    └── before-reading.md
```

需要具体格式或示例时再读对应文件，避免一次性塞满上下文。

## 使用场景

在以下情况触发此技能：

| 场景 | 示例 |
|------|------|
| 记忆写入 | 「把今天做的事情记下来」「读完这篇论文后存入记忆」 |
| 记忆检索 | 「之前关于 XX 的讨论内容是什么」「上周我在做什么」 |
| 服务部署 | 「启动 TMG」「帮我配置记忆服务」 |
| 身份集成 | 「把记忆能力写进 SOUL.md」「配置心跳记忆同步」 |

## 系统简介

- TMG 是**统一的自然语言记忆图**，不是多库/多标签系统
- 所有记忆写入同一张图，系统自动完成概念抽取、关系构建、语义对齐
- 系统只负责 **Remember**（写入）和 **Find**（检索）
- **Select**（筛选与决策）由调用方（你）完成
- 每条记忆带时间戳，实体/关系有版本链，支持时间回溯

## 项目信息

| 项 | 值 |
|----|-----|
| 仓库 | `https://github.com/ngyygm/Temporal_Memory_Graph` |
| 项目根目录 | `Temporal_Memory_Graph/` |
| 依赖文件 | `requirements.txt` |
| 配置文件 | `service_config.json`（模板：`service_config.example.json`） |
| 默认地址 | `http://127.0.0.1:16200` |
| 健康检查 | `GET /health` |
| 启动命令 | `python service_api.py --config service_config.json` |

## 部署与启动

按顺序执行，已完成的步骤可跳过。

### 1. 克隆仓库（如本地不存在）

```bash
git clone https://github.com/ngyygm/Temporal_Memory_Graph
cd Temporal_Memory_Graph
```""",
        "source_name": "并发测试-短",
        "event_time": "2026-03-09T16:01:00",
    },
]
delay_between_submit_seconds = 3


def example_remember_concurrent():
    """按列表 CONCURRENT_TASKS 依次提交：每项间隔 delay_between_submit_seconds 秒，再轮询全部直到完成。
    时间轴以请求中的 event_time 为准，与完成先后无关。"""
    print("\n>>> Remember 并发测试：列表驱动，隔几秒唤起一个")
    n = len(CONCURRENT_TASKS)
    print(f"  共 {n} 条数据，每条间隔 {delay_between_submit_seconds}s 提交")

    # 1) 按列表顺序提交，每条隔几秒
    submitted = []
    for i, spec in enumerate(CONCURRENT_TASKS):
        if i > 0:
            print(f"\n  等待 {delay_between_submit_seconds}s 后提交第 {i + 1} 条…")
            time.sleep(delay_between_submit_seconds)
        label = spec.get("label", f"任务{i+1}")
        t0 = time.time()
        resp = _remember_get(
            spec["text"],
            source_name=spec["source_name"],
            event_time=spec["event_time"],
        )
        elapsed = time.time() - t0
        pp(f"{label} 提交", resp)
        print(f"  {label} 提交耗时: {_fmt_sec(elapsed)}")
        data = resp.json().get("data", {})
        task_id = data.get("task_id")
        if task_id:
            submitted.append((label, task_id))
        else:
            print(f"  {label} 未返回 task_id，跳过")
    if not submitted:
        print("没有成功提交任何任务，跳过并发测试。")
        return
    _print_queue_snapshot()

    # 2) 轮询全部直到都结束
    def get_status(tid):
        r = requests.get(f"{API_BASE}/api/remember/status/{tid}")
        return r.json().get("data", {})

    deadline = time.time() + 600
    interval = 1
    poll_count = 0
    results = {tid: None for _, tid in submitted}
    done = {tid: False for _, tid in submitted}

    while time.time() < deadline:
        poll_count += 1
        for _, tid in submitted:
            if not done[tid]:
                data = get_status(tid)
                results[tid] = data
                if data.get("status") in ("completed", "failed"):
                    done[tid] = True
        statuses = " ".join(f"{label}={results[tid].get('status', '?')}" for label, tid in submitted)
        print(f"  … 轮询[{poll_count}] {statuses}")
        if all(done.values()):
            break
        time.sleep(interval)
        interval = min(interval * 2, 10)

    # 3) 汇总
    print("\n" + "=" * 60)
    print("  并发测试结果汇总")
    print("=" * 60)
    for label, tid in submitted:
        _print_task_timing(label, results.get(tid))
    _print_queue_snapshot()

    completed = sum(1 for _, tid in submitted if (results.get(tid) or {}).get("status") == "completed")
    failed = sum(1 for _, tid in submitted if (results.get(tid) or {}).get("status") == "failed")
    if failed:
        print(f"  ⚠ {failed} 个任务失败")
    if completed == len(submitted):
        print(f"  ✓ 全部 {len(submitted)} 个任务已完成。")
    else:
        print(f"  完成 {completed}/{len(submitted)} 个任务。")


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
