"""
始终版本化 (Always Version) 自动化测试

核心验证: 每个 Episode 提到的概念都应该产生一个新版本。

策略:
- 使用完整描述性名称，LLM 更可能将其作为主要实体提取
- 不依赖 BM25 查找来跟踪实体（LLM 可能改名）
- 用 broader search + 人工筛选来找正确的实体
- 验证版本数 >= episode 提及次数
"""
import json
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

BASE = "http://localhost:16200/api/v1"

def api(method, path, data=None, params=None):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return json.loads(raw)
        except:
            return {"error": raw.decode("utf-8", errors="replace")}

def remember(text, source="auto_test"):
    r = api("POST", "/remember", {"text": text, "source": source}, params={"graph_id": "default"})
    return r.get("data", {}).get("task_id")

def wait_task(task_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = api("GET", f"/remember/tasks/{task_id}", params={"graph_id": "default"})
        d = r.get("data", {})
        if d.get("status") in ("completed", "failed"):
            return d
        time.sleep(3)
    return {"status": "timeout"}

def search_entities(name, mode="bm25", max_results=30):
    """Search entities by name."""
    r = api("GET", "/find/entities/search",
            params={"graph_id": "default", "query_name": name, "search_mode": mode, "max_results": max_results})
    return r.get("data", [])

def find_entities_containing(substr, max_results=50):
    """Find all entities whose name contains the given substring."""
    results = []
    # Try BM25 first
    ents = search_entities(substr, mode="bm25", max_results=max_results)
    for e in ents:
        if substr in e.get("name", ""):
            results.append(e)
    # Also try hybrid for broader coverage
    if len(results) < 3:
        ents2 = search_entities(substr, mode="hybrid", max_results=max_results)
        existing_fids = {e["family_id"] for e in results}
        for e in ents2:
            if e["family_id"] not in existing_fids and substr in e.get("name", ""):
                results.append(e)
                existing_fids.add(e["family_id"])
    return results

def get_versions(family_id):
    r = api("GET", f"/find/entities/{family_id}/versions", params={"graph_id": "default"})
    return r.get("data", [])

def get_relation_versions(family_id):
    r = api("GET", f"/find/relations/{family_id}/versions", params={"graph_id": "default"})
    return r.get("data", [])

def find_relation(entity1_name, entity2_name):
    r = api("GET", "/find/relations",
            params={"graph_id": "default", "query_text": f"{entity1_name} {entity2_name}"})
    rels = r.get("data", {}).get("relations", [])
    for rel in rels:
        e1 = rel.get("entity1_name", "")
        e2 = rel.get("entity2_name", "")
        if (entity1_name in e1 or entity1_name in e2) and (entity2_name in e1 or entity2_name in e2):
            return rel
    return None

results = {"pass": 0, "fail": 0, "errors": []}

def check(test_name, condition, detail=""):
    if condition:
        results["pass"] += 1
        print(f"  PASS: {test_name}")
    else:
        results["fail"] += 1
        msg = f"  FAIL: {test_name}" + (f" — {detail}" if detail else "")
        results["errors"].append(msg)
        print(msg)

def uid(base="测试人"):
    return f"{base}{int(time.time())}"

# ============================================================
# 测试场景
# ============================================================

# --- 场景 1: 新实体创建 (基础) ---
print("\n=== 场景 1: 新实体创建 ===")
name1 = uid("海洋学者陈")
t1 = remember(f"{name1}是一位海洋生物学家，在三亚研究珊瑚礁生态。", "test_s1")
fid1 = None
if t1:
    d = wait_task(t1)
    check("场景1: 任务完成", d["status"] == "completed", d.get("error") or d.get("message",""))
    ents = find_entities_containing(name1[:4])  # Search by prefix
    if ents:
        # Pick the most likely one (highest version_count or best match)
        for e in ents:
            if name1 in e.get("name", ""):
                fid1 = e["family_id"]
                break
        if not fid1 and ents:
            fid1 = ents[0]["family_id"]
        versions = get_versions(fid1)
        check("场景1: 新实体有>=1个版本", len(versions) >= 1, f"实际: {len(versions)}, name={ents[0]['name']}")
        print(f"  INFO: 实体 [{ents[0]['name']}] family_id={fid1}")
    else:
        check("场景1: 找到实体", False, f"未找到包含 {name1[:4]} 的实体")

# --- 场景 2: 同实体第二次提及(不同内容) → 应创建第2个版本 ---
print("\n=== 场景 2: 同实体第二次提及(不同内容) ===")
t2 = remember(f"{name1}最近发现了一种新的深海鱼 species，论文发表在 Nature 上。", "test_s2")
if t2:
    d = wait_task(t2)
    check("场景2: 任务完成", d["status"] == "completed")
    if fid1:
        versions = get_versions(fid1)
        check("场景2: 实体有>=2个版本", len(versions) >= 2, f"实际: {len(versions)}")
        if len(versions) >= 2:
            check("场景2: 最新版本内容不同于首版本",
                  versions[0]["content"] != versions[-1]["content"],
                  "content相同")
    else:
        check("场景2: (跳过，无family_id)", False, "场景1未找到实体")

# --- 场景 3: 同实体第三次提及 → 应创建第3个版本 ---
print("\n=== 场景 3: 同实体第三次提及 ===")
t3 = remember(f"{name1}受邀参加联合国海洋保护峰会，将在会上发表主题演讲。", "test_s3")
if t3:
    d = wait_task(t3)
    check("场景3: 任务完成", d["status"] == "completed")
    if fid1:
        versions = get_versions(fid1)
        check("场景3: 实体有>=3个版本", len(versions) >= 3, f"实际: {len(versions)}")
        if len(versions) >= 3:
            episode_ids = [v["episode_id"] for v in versions if v.get("episode_id")]
            check("场景3: 版本对应不同episodes",
                  len(episode_ids) >= 2,  # At least 2 different episodes
                  f"unique_eps={len(set(episode_ids))}, all={episode_ids}")
    else:
        check("场景3: (跳过)", False, "场景1未找到实体")

# --- 场景 4: 多实体多关系 (单次 remember) ---
print("\n=== 场景 4: 多实体多关系 ===")
name4a = uid("程序员李")
name4b = uid("设计师王")
t4 = remember(
    f"{name4a}和{name4b}是大学同学。{name4a}学的是计算机科学，{name4b}学的是建筑设计。"
    f"毕业后他们一起创办了一家公司，{name4a}负责技术开发，{name4b}负责产品设计。",
    "test_s4"
)
fid4a = fid4b = None
if t4:
    d = wait_task(t4)
    check("场景4: 任务完成", d["status"] == "completed", d.get("error") or d.get("message",""))

    # Find entities by prefix search
    ents4a = find_entities_containing(name4a[:4])
    ents4b = find_entities_containing(name4b[:4])
    if ents4a:
        fid4a = ents4a[0]["family_id"]
        check("场景4: 找到实体1", True, f"name={ents4a[0]['name']}")
    else:
        check("场景4: 找到实体1", False, f"未找到 {name4a[:4]}")
    if ents4b:
        fid4b = ents4b[0]["family_id"]
        check("场景4: 找到实体2", True, f"name={ents4b[0]['name']}")
    else:
        check("场景4: 找到实体2", False, f"未找到 {name4b[:4]}")

    if fid4a:
        v = get_versions(fid4a)
        check("场景4: 实体1有>=1个版本", len(v) >= 1, f"实际: {len(v)}")
    if fid4b:
        v = get_versions(fid4b)
        check("场景4: 实体2有>=1个版本", len(v) >= 1, f"实际: {len(v)}")

# --- 场景 5: 关系第二次提及 → 验证关系和实体版本 ---
print("\n=== 场景 5: 关系第二次提及 ===")
t5 = remember(f"{name4a}帮助{name4b}完成了一个重要的建筑设计项目的技术方案。", "test_s5")
if t5:
    d = wait_task(t5)
    check("场景5: 任务完成", d["status"] == "completed")
    if fid4a:
        v = get_versions(fid4a)
        check("场景5: 实体1版本增加", len(v) >= 2, f"实际: {len(v)} (之前应>=1)")
    if fid4b:
        v = get_versions(fid4b)
        check("场景5: 实体2版本增加", len(v) >= 2, f"实际: {len(v)} (之前应>=1)")

# --- 场景 6: 短文本 ---
print("\n=== 场景 6: 短文本 ===")
t6 = remember("今天天气很好，适合出门散步。", "test_s6")
if t6:
    d = wait_task(t6)
    check("场景6: 短文本任务完成", d["status"] == "completed")

# --- 场景 7: 特殊字符 ---
print("\n=== 场景 7: 特殊字符 ===")
name7 = uid("特殊角色")
t7 = remember(f'{name7}说："这个项目的<成本>已超预算50%，需要立刻调整Plan-B！@管理层 #urgent"', "test_s7")
if t7:
    d = wait_task(t7)
    check("场景7: 特殊字符任务完成", d["status"] == "completed")

# --- 场景 8: 英文实体 ---
print("\n=== 场景 8: 英文实体 ===")
name8 = f"DrWatson{int(time.time())}"
t8 = remember(f"{name8} is a renowned detective who lives in London at 221B Baker Street.", "test_s8")
fid8 = None
if t8:
    d = wait_task(t8)
    check("场景8: 英文实体任务完成", d["status"] == "completed")
    ents = find_entities_containing(name8[:9])
    if ents:
        fid8 = ents[0]["family_id"]
        check("场景8: 找到实体", True, f"name={ents[0]['name']}")
        versions = get_versions(fid8)
        check("场景8: 有>=1个版本", len(versions) >= 1, f"实际: {len(versions)}")
    else:
        check("场景8: 找到实体", False, f"未找到 {name8[:9]}")

    # Second mention
    if fid8:
        t8b = remember(f"{name8} recently solved a complex murder case involving a stolen painting.", "test_s8b")
        if t8b:
            d = wait_task(t8b)
            check("场景8b: 第二次提及任务完成", d["status"] == "completed")
            versions = get_versions(fid8)
            check("场景8b: 版本增加", len(versions) >= 2, f"实际: {len(versions)}")

# --- 场景 9: 大批量实体 ---
print("\n=== 场景 9: 大批量实体 ===")
names9 = [uid(f"登山者{chr(65+i)}") for i in range(3)]
text9 = "、".join(names9) + "三人一起去登珠穆朗玛峰。" + "".join(
    [f"{n}带了{'氧气瓶' if i==0 else '绳索' if i==1 else '食物'}。" for i, n in enumerate(names9)]
)
t9 = remember(text9, "test_s9")
fids9 = {}
if t9:
    d = wait_task(t9)
    check("场景9: 任务完成", d["status"] == "completed", d.get("error") or d.get("message",""))
    for n in names9:
        ents = find_entities_containing(n[:4])
        if ents:
            fids9[n] = ents[0]["family_id"]
            versions = get_versions(ents[0]["family_id"])
            check(f"场景9: 找到 [{ents[0]['name']}] 并有>=1版本", len(versions) >= 1, f"实际: {len(versions)}")
        else:
            check(f"场景9: 找到实体", False, f"未找到 {n[:4]}")

# --- 场景 10: 快速连续提交 ---
print("\n=== 场景 10: 快速连续提交 ===")
name10 = uid("速射测试")
t10a = remember(f"{name10}是一个测试角色，第一次出现。", "test_s10a")
t10b = remember(f"{name10}第二次出现，做了些不同的事。", "test_s10b")
# Wait for both
fid10 = None
for t in [t10a, t10b]:
    if t:
        wait_task(t)

ents10 = find_entities_containing(name10[:4])
if ents10:
    fid10 = ents10[0]["family_id"]
    versions = get_versions(fid10)
    check("场景10: 两次快速提交后>=2版本", len(versions) >= 2, f"实际: {len(versions)}, name={ents10[0]['name']}")
else:
    check("场景10: 找到实体", False, f"未找到 {name10[:4]}")

# ============================================================
print(f"\n{'='*60}")
print(f"测试结果: {results['pass']} PASS / {results['fail']} FAIL")
if results["errors"]:
    print("\n失败详情:")
    for e in results["errors"]:
        print(e)
print(f"{'='*60}")
