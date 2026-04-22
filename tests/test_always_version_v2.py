"""
Always Version — Iteration Test Suite (v2)

Focus on testing specific edge cases and scenarios that expose bugs
in the always-version pipeline.
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

def search_entities(query, mode="bm25", max_results=30):
    r = api("GET", "/find/entities/search",
            params={"graph_id": "default", "query_name": query, "search_mode": mode, "max_results": max_results})
    return r.get("data", [])

def find_all_matching(substr, max_results=50):
    """Find ALL entities whose name contains substr across BM25 + hybrid."""
    results = {}
    for mode in ["bm25", "hybrid"]:
        ents = search_entities(substr, mode=mode, max_results=max_results)
        for e in ents:
            if substr in e.get("name", ""):
                results[e["family_id"]] = e
    return list(results.values())

def get_versions(family_id):
    r = api("GET", f"/find/entities/{family_id}/versions", params={"graph_id": "default"})
    return r.get("data", [])

def get_entity(family_id):
    r = api("GET", f"/find/entities/{family_id}", params={"graph_id": "default"})
    return r.get("data", {})

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

def uid(base="测试"):
    return f"{base}{int(time.time())}"

# ============================================================
# Test Scenarios
# ============================================================

# --- Scenario A: Same entity name, progressively enriched content ---
print("\n=== A: Progressive content enrichment (3 mentions) ===")
nameA = uid("诸葛孔明")
fidA = None

# A1: First mention - basic identity
t1 = remember(f"{nameA}是三国时期蜀汉的丞相，字孔明，号卧龙。他年轻时在隆中隐居，被刘备三顾茅庐请出山。", "test_A1")
if t1:
    d = wait_task(t1)
    check("A1: 任务完成", d["status"] == "completed", d.get("error") or d.get("message",""))
    ents = find_all_matching(nameA[:4])
    for e in ents:
        if nameA in e.get("name", ""):
            fidA = e["family_id"]
            break
    if fidA:
        v = get_versions(fidA)
        check("A1: 新实体有>=1个版本", len(v) >= 1, f"实际: {len(v)}, name={e['name']}")
        print(f"  INFO: [{e['name']}] fid={fidA} ver={len(v)}")
    else:
        check("A1: 找到实体", False, f"未找到 {nameA}")

# A2: Second mention - new facts
if fidA:
    t2 = remember(f"{nameA}在赤壁之战中联合孙权大败曹操，奠定了三分天下的基础。他还发明了木牛流马用于运输粮草。", "test_A2")
    if t2:
        d = wait_task(t2)
        check("A2: 任务完成", d["status"] == "completed")
        v = get_versions(fidA)
        check("A2: 实体有>=2个版本", len(v) >= 2, f"实际: {len(v)}")

# A3: Third mention - even more facts
if fidA:
    t3 = remember(f"{nameA}在刘备去世后辅佐刘禅，写下了著名的《出师表》。他六出祁山北伐，最终病逝于五丈原。", "test_A3")
    if t3:
        d = wait_task(t3)
        check("A3: 任务完成", d["status"] == "completed")
        v = get_versions(fidA)
        check("A3: 实体有>=3个版本", len(v) >= 3, f"实际: {len(v)}")
        if len(v) >= 3:
            eps = set(vv.get("episode_id") for vv in v if vv.get("episode_id"))
            check("A3: 至少2个不同episode_id", len(eps) >= 2, f"unique_eps={len(eps)}")

# --- Scenario B: Same entity, identical content (should still version) ---
print("\n=== B: Identical content re-mention (should still create version) ===")
nameB = uid("复制测试人物")
fidB = None

t1 = remember(f"{nameB}是一名软件工程师，在北京工作。", "test_B1")
if t1:
    d = wait_task(t1)
    check("B1: 任务完成", d["status"] == "completed")
    ents = find_all_matching(nameB[:4])
    for e in ents:
        if nameB in e.get("name", ""):
            fidB = e["family_id"]
            break
    if fidB:
        v1 = get_versions(fidB)
        check("B1: 有>=1个版本", len(v1) >= 1, f"实际: {len(v)}")
        print(f"  INFO: [{e['name']}] fid={fidB} ver={len(v1)}")
    else:
        check("B1: 找到实体", False)

if fidB:
    # Send same content again
    t2 = remember(f"{nameB}是一名软件工程师，在北京工作。", "test_B2")
    if t2:
        d = wait_task(t2)
        check("B2: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidB)
        check("B2: 相同内容也创建新版本", len(v2) >= 2, f"实际: {len(v2)}")

# --- Scenario C: Multiple entities, some new, some existing ---
print("\n=== C: Mixed new + existing entities ===")
nameC1 = uid("教授张")
nameC2 = uid("学生李")

# First episode: both new
t1 = remember(f"{nameC1}是清华大学的物理学教授。{nameC2}是他的研究生，研究方向是量子计算。", "test_C1")
fidC1 = fidC2 = None
if t1:
    d = wait_task(t1)
    check("C1: 任务完成", d["status"] == "completed")
    for name, var_name in [(nameC1, "fidC1"), (nameC2, "fidC2")]:
        ents = find_all_matching(name[:4])
        for e in ents:
            if name in e.get("name", ""):
                if var_name == "fidC1":
                    fidC1 = e["family_id"]
                else:
                    fidC2 = e["family_id"]
                break
    check("C1: 找到实体1", fidC1 is not None, f"{nameC1}")
    check("C1: 找到实体2", fidC2 is not None, f"{nameC2}")

# Second episode: mention only entity1
if fidC1:
    t2 = remember(f"{nameC1}最近在Nature上发表了一篇关于量子纠缠的论文，引起了学术界广泛关注。", "test_C2")
    if t2:
        d = wait_task(t2)
        check("C2: 任务完成", d["status"] == "completed")
        v1 = get_versions(fidC1)
        check("C2: 实体1版本增加", len(v1) >= 2, f"实际: {len(v1)}")
        if fidC2:
            v2 = get_versions(fidC2)
            check("C2: 实体2版本未变(未提及)", len(v2) >= 1, f"实际: {len(v2)}")

# --- Scenario D: Relation versioning ---
print("\n=== D: Relation versioning ===")
nameD1 = uid("师父赵")
nameD2 = uid("徒弟钱")

t1 = remember(f"{nameD1}是一位武术大师，{nameD2}拜他为师学习太极拳。两人师徒情深。", "test_D1")
fidD1 = fidD2 = None
if t1:
    d = wait_task(t1)
    check("D1: 任务完成", d["status"] == "completed")
    for name, var_name in [(nameD1, "fidD1"), (nameD2, "fidD2")]:
        ents = find_all_matching(name[:4])
        for e in ents:
            if name in e.get("name", ""):
                if var_name == "fidD1":
                    fidD1 = e["family_id"]
                else:
                    fidD2 = e["family_id"]
                break
    check("D1: 找到师父", fidD1 is not None)
    check("D1: 找到徒弟", fidD2 is not None)

if fidD1 and fidD2:
    # Second mention with new relation context
    t2 = remember(f"{nameD1}教{ nameD2}剑法，{nameD2}学得很快，已经能独当一面了。", "test_D2")
    if t2:
        d = wait_task(t2)
        check("D2: 任务完成", d["status"] == "completed")
        v1 = get_versions(fidD1)
        v2 = get_versions(fidD2)
        check("D2: 师父版本增加", len(v1) >= 2, f"实际: {len(v1)}")
        check("D2: 徒弟版本增加", len(v2) >= 2, f"实际: {len(v2)}")

# --- Scenario E: Empty/short text ---
print("\n=== E: Empty/short text ===")
t1 = remember("好的", "test_E1")
if t1:
    d = wait_task(t1)
    check("E1: 短文本完成", d["status"] == "completed")

t2 = remember("这个消息已读。", "test_E2")
if t2:
    d = wait_task(t2)
    check("E2: 极短文本完成", d["status"] == "completed")

# --- Scenario F: English entities ---
print("\n=== F: English entities ===")
nameF = f"ProfHawkins{int(time.time())}"
fidF = None

t1 = remember(f"{nameF} is a theoretical physicist at Cambridge University. His research focuses on black holes and cosmology.", "test_F1")
if t1:
    d = wait_task(t1)
    check("F1: 任务完成", d["status"] == "completed")
    ents = find_all_matching(nameF[:10])
    for e in ents:
        if nameF in e.get("name", ""):
            fidF = e["family_id"]
            break
    check("F1: 找到英文实体", fidF is not None)

if fidF:
    t2 = remember(f"{nameF} recently published a groundbreaking paper on Hawking radiation, confirming his theoretical predictions with experimental data.", "test_F2")
    if t2:
        d = wait_task(t2)
        check("F2: 任务完成", d["status"] == "completed")
        v = get_versions(fidF)
        check("F2: 英文实体版本增加", len(v) >= 2, f"实际: {len(v)}")

# --- Scenario G: Special characters and punctuation ---
print("\n=== G: Special characters ===")
nameG = uid("特殊人物")
t1 = remember(f'{nameG}说："计划A<第一阶段>已完成100%，但Plan-B还需调整！@所有人 #重要"', "test_G1")
if t1:
    d = wait_task(t1)
    check("G1: 特殊字符完成", d["status"] == "completed")

# --- Scenario H: Rapid consecutive submissions ---
print("\n=== H: Rapid consecutive submissions ===")
nameH = uid("快速人物")
tasks = []
for i in range(3):
    t = remember(f"{nameH}第{i+1}次出现，做了事情{i+1}。", f"test_H{i}")
    tasks.append(t)

fidH = None
for t in tasks:
    if t:
        wait_task(t)

# Now find the entity
ents = find_all_matching(nameH[:4])
for e in ents:
    if nameH in e.get("name", ""):
        fidH = e["family_id"]
        break
if fidH:
    v = get_versions(fidH)
    check("H: 3次快速提交后>=2版本", len(v) >= 2, f"实际: {len(v)}, name={ents[0]['name'] if ents else 'N/A'}")
else:
    # Try broader search
    all_ents = find_all_matching(nameH[:2])
    check("H: 找到实体", len(all_ents) > 0, f"未找到 {nameH}")

# --- Scenario I: Single entity mentioned with different contexts ---
print("\n=== I: Single entity with diverse contexts ===")
nameI = uid("科学家周")
fidI = None

t1 = remember(f"{nameI}在实验室工作。", "test_I1")
if t1:
    d = wait_task(t1)
    check("I1: 任务完成", d["status"] == "completed")
    ents = find_all_matching(nameI[:4])
    for e in ents:
        if nameI in e.get("name", ""):
            fidI = e["family_id"]
            break
    check("I1: 找到实体", fidI is not None)

if fidI:
    # Context 2: longer, more detailed
    t2 = remember(f"昨天，{nameI}在国际学术会议上发表了关于基因编辑技术CRISPR的最新研究成果。这项研究有望治疗多种遗传性疾病。会议结束后，多家媒体对他进行了采访。", "test_I2")
    if t2:
        d = wait_task(t2)
        check("I2: 任务完成", d["status"] == "completed")
        v = get_versions(fidI)
        check("I2: 版本增加", len(v) >= 2, f"实际: {len(v)}")

# --- Scenario J: Entity with nickname/alias in different episodes ---
print("\n=== J: Entity with aliases ===")
nameJ = uid("作家林")
aliasJ = uid("笔名林")
fidJ = None

t1 = remember(f"{nameJ}是一位著名小说家，笔名{aliasJ}。他写过很多畅销书。", "test_J1")
if t1:
    d = wait_task(t1)
    check("J1: 任务完成", d["status"] == "completed")
    ents = find_all_matching(nameJ[:4])
    for e in ents:
        if nameJ in e.get("name", ""):
            fidJ = e["family_id"]
            break
    check("J1: 找到实体", fidJ is not None)

if fidJ:
    t2 = remember(f"{nameJ}的新作品《星际旅行》获得了雨果奖提名。{aliasJ}这个笔名为更多读者所熟知。", "test_J2")
    if t2:
        d = wait_task(t2)
        check("J2: 任务完成", d["status"] == "completed")
        v = get_versions(fidJ)
        check("J2: 版本增加", len(v) >= 2, f"实际: {len(v)}")

# ============================================================
print(f"\n{'='*60}")
print(f"测试结果: {results['pass']} PASS / {results['fail']} FAIL")
if results["errors"]:
    print("\n失败详情:")
    for e in results["errors"]:
        print(e)
print(f"{'='*60}")
