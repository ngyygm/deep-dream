"""
使用示例：通过 HTTP API 操作 Temporal Memory Graph

前置条件：
  1. 启动 API 服务：python service_api.py --config service_config.json
  2. 确保测试文档存在：../datas/docs/三体2黑暗森林.txt

本脚本演示三个核心场景：
  1. Remember — 传文本记忆
  2. Remember — 传文件记忆
  3. Find — 语义检索唤醒局部记忆
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


# ------------------------------------------------------------------
# 1. Remember — 传文本
# ------------------------------------------------------------------
def example_remember_text():
    print("\n>>> Remember: 传文本记忆")

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
        json={"text": text, "source_name": "三体测试-文本"},
    )
    pp("Remember Text", resp)
    return resp.json()


# ------------------------------------------------------------------
# 2. Remember — 传本地文件路径
# ------------------------------------------------------------------
def example_remember_file():
    print("\n>>> Remember: 传本地文件路径")

    file_path = "/home/linkco/exa/datas/docs/三体2黑暗森林.txt"
    resp = requests.post(
        f"{API_BASE}/api/remember",
        json={"file_path": file_path, "source_name": "三体2黑暗森林"},
    )
    pp("Remember File", resp)
    return resp.json()


# ------------------------------------------------------------------
# 3. Remember — 上传文件
# ------------------------------------------------------------------
def example_remember_upload():
    print("\n>>> Remember: 上传文件")

    file_path = "/home/linkco/exa/datas/docs/三体2黑暗森林.txt"
    with open(file_path, "rb") as f:
        # 只取前 2000 字符作为演示，避免处理时间过长
        content = f.read(2000)

    resp = requests.post(
        f"{API_BASE}/api/remember",
        files={"file": ("三体2片段.txt", content, "text/plain")},
        data={"source_name": "三体2黑暗森林-上传片段"},
    )
    pp("Remember Upload", resp)
    return resp.json()


# ------------------------------------------------------------------
# 4. Find — 统一语义检索
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
# 5. Find — 原子接口示例
# ------------------------------------------------------------------
def example_find_atomic():
    print("\n>>> Find: 原子接口")

    # 统计
    resp = requests.get(f"{API_BASE}/api/find/stats")
    pp("Stats", resp)

    # 搜索实体
    resp = requests.get(
        f"{API_BASE}/api/find/entities/search",
        params={"query_name": "罗辑", "max_results": 5, "threshold": 0.3},
    )
    pp("Entity Search: 罗辑", resp)

    # 搜索关系
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

    # 选择要运行的示例
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("text", "all"):
        example_remember_text()

    if mode in ("file", "all"):
        example_remember_file()

    if mode in ("upload", "all"):
        example_remember_upload()

    if mode in ("find", "all"):
        example_find()

    if mode in ("atomic", "all"):
        example_find_atomic()

    if mode not in ("text", "file", "upload", "find", "atomic", "all"):
        print(f"""
用法: python example_usage.py [mode]

mode 可选值:
  text    — 仅测试文本记忆
  file    — 仅测试文件路径记忆（三体2黑暗森林.txt）
  upload  — 仅测试文件上传记忆
  find    — 仅测试语义检索
  atomic  — 仅测试原子接口
  all     — 运行全部示例（默认）
""")
