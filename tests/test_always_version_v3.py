"""
Always Version — Test Suite v3

Strategy: Use episode-to-entity tracking to find extracted entities,
avoiding LLM rename issues. Each test:
1. Remembers text
2. Finds the episode
3. Gets entities linked to that episode
4. Checks version counts
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

def get_versions(family_id):
    r = api("GET", f"/find/entities/{family_id}/versions", params={"graph_id": "default"})
    return r.get("data", [])

def find_episode_by_content(substr, limit=20):
    """Search episodes by content substring."""
    r = api("GET", "/episodes", params={"graph_id": "default", "limit": limit})
    eps = r.get("data", {}).get("episodes", [])
    for ep in eps:
        if substr in ep.get("content", ""):
            return ep
    return None

def get_episode_entities(episode_uuid):
    """Get entities linked to an episode."""
    r = api("GET", f"/episodes/{urllib.parse.quote(episode_uuid, safe='')}/entities",
            params={"graph_id": "default"})
    return r.get("data", {}).get("entities", [])

def find_entity_family_id_by_name_substring(substr, episode_uuid=None):
    """Find entity family_id by name substring, optionally from a specific episode."""
    if episode_uuid:
        ents = get_episode_entities(episode_uuid)
        for e in ents:
            if substr in e.get("name", ""):
                return e["family_id"], e["name"]
    # Fallback: search all entities
    r = api("GET", "/find/entities/search",
            params={"graph_id": "default", "query_name": substr, "search_mode": "hybrid", "max_results": 20})
    for e in r.get("data", []):
        if substr in e.get("name", ""):
            return e["family_id"], e["name"]
    return None, None

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

# --- A: Basic 3-mention progressive enrichment ---
print("\n=== A: Progressive enrichment (3 mentions, same entity) ===")
nameA = uid("孔明先生")
markerA = uid("MK_A")

t1 = remember(f"{nameA}是一位智者。{markerA}", "test_A1")
fidA = None
if t1:
    d = wait_task(t1)
    check("A1: 任务完成", d["status"] == "completed")
    # Find entity from the episode
    fidA, actual_name = find_entity_family_id_by_name_substring(nameA[:4])
    if not fidA:
        # Try searching by marker
        fidA, actual_name = find_entity_family_id_by_name_substring(markerA[:6])
    check("A1: 找到实体", fidA is not None, f"搜索: {nameA[:4]}")
    if fidA:
        v = get_versions(fidA)
        check("A1: >=1版本", len(v) >= 1, f"实际: {len(v)}, name={actual_name}")
        print(f"  INFO: name={actual_name} fid={fidA} ver={len(v)}")

if fidA:
    t2 = remember(f"{nameA}最近在赤壁用火攻大败敌军。他的智谋令人叹服。", "test_A2")
    if t2:
        d = wait_task(t2)
        check("A2: 任务完成", d["status"] == "completed")
        v = get_versions(fidA)
        check("A2: >=2版本", len(v) >= 2, f"实际: {len(v)}")

if fidA:
    t3 = remember(f"{nameA}写下了出师表，表达了对国家的忠诚。他六出祁山，鞠躬尽瘁。", "test_A3")
    if t3:
        d = wait_task(t3)
        check("A3: 任务完成", d["status"] == "completed")
        v = get_versions(fidA)
        check("A3: >=3版本", len(v) >= 3, f"实际: {len(v)}")
        if len(v) >= 3:
            eps = set(vv.get("episode_id") for vv in v if vv.get("episode_id"))
            check("A3: >=2 unique episode_ids", len(eps) >= 2, f"unique={len(eps)}")

# --- B: Identical content → should still create version ---
print("\n=== B: Identical content re-mention ===")
nameB = uid("克隆测试")
t1 = remember(f"{nameB}是一只在实验室长大的小白鼠。{uid('MB')}", "test_B1")
fidB = None
if t1:
    d = wait_task(t1)
    check("B1: 任务完成", d["status"] == "completed")
    fidB, actual_name = find_entity_family_id_by_name_substring(nameB[:4])
    check("B1: 找到实体", fidB is not None, f"搜索: {nameB[:4]}")
    if fidB:
        v1 = get_versions(fidB)
        check("B1: >=1版本", len(v1) >= 1, f"实际: {len(v1)}")
        print(f"  INFO: name={actual_name} fid={fidB} ver={len(v1)}")

if fidB:
    t2 = remember(f"{nameB}是一只在实验室长大的小白鼠。", "test_B2")
    if t2:
        d = wait_task(t2)
        check("B2: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidB)
        check("B2: 相同内容也创建新版本", len(v2) >= 2, f"实际: {len(v2)}")

# --- C: English entity ---
print("\n=== C: English entity ===")
nameC = f"DrAlice{int(time.time())}"
t1 = remember(f"Dr. {nameC} is a marine biologist studying coral reefs in the Pacific Ocean.", "test_C1")
fidC = None
if t1:
    d = wait_task(t1)
    check("C1: 任务完成", d["status"] == "completed")
    fidC, actual_name = find_entity_family_id_by_name_substring(nameC[:7])
    check("C1: 找到英文实体", fidC is not None)
    if fidC:
        v1 = get_versions(fidC)
        check("C1: >=1版本", len(v1) >= 1)
        print(f"  INFO: name={actual_name} fid={fidC} ver={len(v1)}")

if fidC:
    t2 = remember(f"Dr. {nameC} recently discovered a new species of deep-sea fish near the Mariana Trench.", "test_C2")
    if t2:
        d = wait_task(t2)
        check("C2: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidC)
        check("C2: 版本增加", len(v2) >= 2, f"实际: {len(v2)}")

# --- D: Multiple entities ---
print("\n=== D: Multiple entities in one episode ===")
nameD1 = uid("老师王")
nameD2 = uid("学生赵")
t1 = remember(f"{nameD1}是{ nameD2}的数学老师。{nameD1}教{ nameD2}学习微积分。两人关系很好。", "test_D1")
fidD1 = fidD2 = None
if t1:
    d = wait_task(t1)
    check("D1: 任务完成", d["status"] == "completed")
    fidD1, n1 = find_entity_family_id_by_name_substring(nameD1[:4])
    fidD2, n2 = find_entity_family_id_by_name_substring(nameD2[:4])
    check("D1: 找到实体1", fidD1 is not None, f"搜索: {nameD1[:4]}")
    check("D1: 找到实体2", fidD2 is not None, f"搜索: {nameD2[:4]}")

if fidD1:
    t2 = remember(f"{nameD1}被评为优秀教师。他的教学方法很受欢迎。", "test_D2")
    if t2:
        d = wait_task(t2)
        check("D2: 任务完成", d["status"] == "completed")
        v1 = get_versions(fidD1)
        check("D2: 实体1版本增加", len(v1) >= 2, f"实际: {len(v1)}")

if fidD2:
    t3 = remember(f"{nameD2}在数学竞赛中获得了一等奖，{nameD1}非常高兴。", "test_D3")
    if t3:
        d = wait_task(t3)
        check("D3: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidD2)
        check("D3: 实体2版本增加", len(v2) >= 2, f"实际: {len(v2)}")
        if fidD1:
            v1 = get_versions(fidD1)
            check("D3: 实体1版本也增加(第二次提及)", len(v1) >= 3, f"实际: {len(v1)}")

# --- E: Rapid consecutive submissions ---
print("\n=== E: Rapid consecutive submissions ===")
nameE = uid("连续提交")
tasks = []
for i in range(3):
    t = remember(f"{nameE}第{i+1}次出现，做事情{i+1}。", f"test_E{i}")
    tasks.append(t)

for t in tasks:
    if t:
        wait_task(t)

fidE, actual_name = find_entity_family_id_by_name_substring(nameE[:4])
check("E: 找到实体", fidE is not None, f"搜索: {nameE[:4]}")
if fidE:
    v = get_versions(fidE)
    check("E: >=2版本", len(v) >= 2, f"实际: {len(v)}, name={actual_name}")

# --- F: Special characters ---
print("\n=== F: Special characters ===")
nameF = uid("特字")
t1 = remember(f'{nameF}说："Plan-A<成本>已超50%！@管理层 #urgent"', "test_F1")
if t1:
    d = wait_task(t1)
    check("F: 特殊字符完成", d["status"] == "completed")

# --- G: Very short text ---
print("\n=== G: Short text ===")
t1 = remember("好的。", "test_G1")
if t1:
    d = wait_task(t1)
    check("G1: 极短文本完成", d["status"] == "completed")

t2 = remember("天晴了。", "test_G2")
if t2:
    d = wait_task(t2)
    check("G2: 短文本完成", d["status"] == "completed")

# --- H: Entity with descriptive name (less likely to be renamed) ---
print("\n=== H: Descriptive entity name ===")
nameH = uid("北京大学计算机系教授")
t1 = remember(f"{nameH}在人工智能领域有很深的研究。{uid('MH')}", "test_H1")
fidH = None
if t1:
    d = wait_task(t1)
    check("H1: 任务完成", d["status"] == "completed")
    fidH, actual_name = find_entity_family_id_by_name_substring(nameH[:6])
    check("H1: 找到实体", fidH is not None)
    if fidH:
        v1 = get_versions(fidH)
        check("H1: >=1版本", len(v1) >= 1)

if fidH:
    t2 = remember(f"{nameH}最近在顶级期刊上发表了关于大语言模型的论文。", "test_H2")
    if t2:
        d = wait_task(t2)
        check("H2: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidH)
        check("H2: 版本增加", len(v2) >= 2, f"实际: {len(v2)}")

# --- I: Entity mentioned alone in second episode ---
print("\n=== I: Entity mentioned alone ===")
nameI = uid("独行侠")
t1 = remember(f"{nameI}是一个神秘的旅行者。{uid('MI')}", "test_I1")
fidI = None
if t1:
    d = wait_task(t1)
    check("I1: 任务完成", d["status"] == "completed")
    fidI, actual_name = find_entity_family_id_by_name_substring(nameI[:4])
    check("I1: 找到实体", fidI is not None)
    if fidI:
        v1 = get_versions(fidI)
        check("I1: >=1版本", len(v1) >= 1)

if fidI:
    # Mention alone in a new episode
    t2 = remember(f"{nameI}昨天到了长安城。", "test_I2")
    if t2:
        d = wait_task(t2)
        check("I2: 任务完成", d["status"] == "completed")
        v2 = get_versions(fidI)
        check("I2: 版本增加(单独提及)", len(v2) >= 2, f"实际: {len(v2)}")

# --- J: Relation versioning ---
print("\n=== J: Relation versioning ===")
nameJ1 = uid("父亲陈")
nameJ2 = uid("儿子陈")
t1 = remember(f"{nameJ1}是{nameJ2}的父亲。{nameJ1}很爱{nameJ2}。{uid('MJ')}", "test_J1")
fidJ1 = fidJ2 = None
if t1:
    d = wait_task(t1)
    check("J1: 任务完成", d["status"] == "completed")
    fidJ1, _ = find_entity_family_id_by_name_substring(nameJ1[:4])
    fidJ2, _ = find_entity_family_id_by_name_substring(nameJ2[:4])
    check("J1: 找到父亲", fidJ1 is not None)
    check("J1: 找到儿子", fidJ2 is not None)

if fidJ1 and fidJ2:
    t2 = remember(f"{nameJ1}教{nameJ2}骑自行车。{nameJ2}学得很快。", "test_J2")
    if t2:
        d = wait_task(t2)
        check("J2: 任务完成", d["status"] == "completed")
        v1 = get_versions(fidJ1)
        v2 = get_versions(fidJ2)
        check("J2: 父亲版本增加", len(v1) >= 2, f"实际: {len(v1)}")
        check("J2: 儿子版本增加", len(v2) >= 2, f"实际: {len(v2)}")

# ============================================================
print(f"\n{'='*60}")
print(f"测试结果: {results['pass']} PASS / {results['fail']} FAIL")
if results["errors"]:
    print("\n失败详情:")
    for e in results["errors"]:
        print(e)
print(f"{'='*60}")
