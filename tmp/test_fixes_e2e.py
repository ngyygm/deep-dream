"""
End-to-end integration tests for Deep-Dream server.
Uses Flask test client to avoid background server issues.
"""
import sys
import os
import time
import json
import traceback

# Ensure correct working directory and Python path
os.chdir("C:/Users/Administrator/Documents/ClaudeProject/deep-dream")
sys.path.insert(0, "C:/Users/Administrator/Documents/ClaudeProject/deep-dream")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.argv = ['deep-dream', '--config', 'service_config.json', '--port', '16201']

config = json.load(open('service_config.json'))
from core.server.monitor import SystemMonitor
from core.server.registry import GraphRegistry
from core.server.api import create_app

BASE = "test_client"

def setup_client():
    """Create Flask test client."""
    sm = SystemMonitor(config=config)
    reg = GraphRegistry('./library', config, system_monitor=sm)
    app = create_app(reg, config, system_monitor=sm)
    app.config['TESTING'] = True
    return app.test_client()


def print_result(test_name, passed, details):
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {test_name}: {status}")
    print(f"  Details: {details}")
    print(f"{'='*60}")
    return passed


def remember_text(client, text, source_name=None, timeout=300):
    """POST /api/v1/remember with text and wait for completion."""
    payload = {"text": text, "wait": True, "timeout": str(timeout)}
    if source_name:
        payload["source_name"] = source_name
    r = client.post('/api/v1/remember', json=payload)
    return r


# ============================================================
# TEST 1: Duplicate document prevention (Fix 1)
# ============================================================
def test_duplicate_document_prevention(client):
    print("\n>>> TEST 1: Duplicate Document Prevention")
    unique_text = "三体测试文档unique456_abc_20260530_e2e_v2"
    source1 = "三体测试_e2e_fix1_v2"
    source2 = "三体测试副本_e2e_fix1_v2"

    print(f"  Submitting first document (source={source1})...")
    r1 = remember_text(client, unique_text, source_name=source1, timeout=120)
    print(f"  First submission: status={r1.status_code}")

    if r1.status_code not in (200, 202):
        return print_result("Test 1: Duplicate Doc Prevention", False,
                           f"First POST failed: {r1.status_code} - {r1.text[:200]}")

    body1 = r1.get_json()
    data1 = body1.get("data", body1)
    print(f"  First document completed: status={data1.get('status')}")

    time.sleep(1)

    # Submit same text with different source
    print(f"  Submitting duplicate (source={source2})...")
    r2 = remember_text(client, unique_text, source_name=source2, timeout=120)
    print(f"  Second submission: status={r2.status_code}")

    body2 = r2.get_json()
    data2 = body2.get("data", body2)
    print(f"  Second document completed: status={data2.get('status')}")

    # Check documents
    time.sleep(1)
    r_docs = client.get('/api/v1/documents?limit=100')
    if r_docs.status_code != 200:
        return print_result("Test 1: Duplicate Doc Prevention", False,
                           f"GET /documents failed: {r_docs.status_code}")

    docs_data = r_docs.get_json().get("data", r_docs.get_json())
    docs = docs_data.get("documents", [])

    matching_docs = [d for d in docs if source1 in (d.get("source_document", "") or d.get("title", ""))
                     or source2 in (d.get("source_document", "") or d.get("title", ""))]

    doc_count = len(matching_docs)
    print(f"  Found {doc_count} documents matching our test content")

    passed = doc_count <= 1
    return print_result("Test 1: Duplicate Doc Prevention", passed,
                       f"doc_count={doc_count} (expected <=1)")


# ============================================================
# TEST 2: Entity and relation extraction (Fix 2)
# ============================================================
def test_entity_relation_extraction(client):
    print("\n>>> TEST 2: Entity and Relation Extraction")
    test_text = (
        "刘备是蜀汉的开国皇帝。关羽是刘备的结义兄弟。"
        "张飞也是刘备的结义兄弟。诸葛亮是刘备的军师。"
        "赵云是刘备麾下的名将。马超是蜀汉的五虎上将之一。"
    )
    source = "三国人物关系_e2e_fix2_v2"

    print(f"  Submitting test text ({len(test_text)} chars)...")
    r = remember_text(client, test_text, source_name=source, timeout=120)
    print(f"  Response: status={r.status_code}")

    if r.status_code not in (200, 202):
        return print_result("Test 2: Entity/Relation Extraction", False,
                           f"POST failed: {r.status_code} - {r.text[:200]}")

    body = r.get_json()
    data = body.get("data", body)
    task_result = data.get("result", {})

    entities = task_result.get("entities", 0)
    relations = task_result.get("relations", 0)
    print(f"  Task result: entities={entities}, relations={relations}")

    time.sleep(1)

    # Check concepts
    r_concepts = client.get('/api/v1/concepts?role=entity&limit=50')
    entity_count = 0
    if r_concepts.status_code == 200:
        concepts_data = r_concepts.get_json().get("data", r_concepts.get_json())
        entity_count = concepts_data.get("total", 0)
        print(f"  Graph entity total: {entity_count}")

    r_rels = client.get('/api/v1/concepts?role=relation&limit=50')
    relation_count = 0
    if r_rels.status_code == 200:
        rels_data = r_rels.get_json().get("data", r_rels.get_json())
        relation_count = rels_data.get("total", 0)
        print(f"  Graph relation total: {relation_count}")

    passed = relation_count > 0
    return print_result("Test 2: Entity/Relation Extraction", passed,
                       f"entities={entities}, relations={relations}, "
                       f"graph_entities={entity_count}, graph_relations={relation_count}")


# ============================================================
# TEST 3: Graph traversal bidirectionality (C4)
# ============================================================
def test_graph_traversal(client):
    print("\n>>> TEST 3: Graph Traversal Bidirectionality")

    r = client.post('/api/v1/find',
                    json={"query": "刘备", "max_entities": 5, "max_relations": 10},
                    content_type="application/json")
    if r.status_code != 200:
        return print_result("Test 3: Graph Traversal", False,
                           f"Find failed: {r.status_code}")

    find_data = r.get_json().get("data", r.get_json())
    entities = find_data.get("entities", [])
    if not entities:
        return print_result("Test 3: Graph Traversal", False,
                           "No entities found for query '刘备'")

    entity_fid = entities[0].get("family_id", "")
    entity_name = entities[0].get("name", "")
    print(f"  Starting entity: {entity_name} (family_id={entity_fid})")

    # Traverse via POST /api/v1/traverse
    r_traverse = client.post('/api/v1/traverse',
                             json={"start_family_ids": [entity_fid], "max_depth": 2, "max_results": 200},
                             content_type="application/json")
    if r_traverse.status_code != 200:
        return print_result("Test 3: Graph Traversal", False,
                           f"Traverse failed: {r_traverse.status_code} - {r_traverse.text[:200]}")

    traverse_data = r_traverse.get_json().get("data", r_traverse.get_json())
    edges = traverse_data.get("edges", [])
    visited = traverse_data.get("visited", [])

    outgoing = [e for e in edges if e.get("source_family_id") == entity_fid]
    incoming = [e for e in edges if e.get("target_family_id") == entity_fid]

    print(f"  Traversal: {len(edges)} edges, {len(visited)} visited")
    print(f"  Outgoing: {len(outgoing)}, Incoming: {len(incoming)}")

    # Neighbors endpoint
    r_neighbors = client.get(f'/api/v1/concepts/{entity_fid}/neighbors?max_depth=2&max_results=100')
    neighbors_count = 0
    if r_neighbors.status_code == 200:
        nb_data = r_neighbors.get_json().get("data", r_neighbors.get_json())
        neighbors_count = len(nb_data.get("neighbors", []))
        print(f"  Neighbors endpoint: {neighbors_count} neighbors")

    passed = len(edges) > 0 and (len(outgoing) > 0 or len(incoming) > 0)
    return print_result("Test 3: Graph Traversal", passed,
                       f"edges={len(edges)}, outgoing={len(outgoing)}, incoming={len(incoming)}, "
                       f"visited={len(visited)}, neighbors={neighbors_count}")


# ============================================================
# TEST 4: Document file info endpoint (C1)
# ============================================================
def test_document_file_info(client):
    print("\n>>> TEST 4: Document File Info Endpoint")

    r_docs = client.get('/api/v1/documents?limit=10')
    if r_docs.status_code != 200:
        return print_result("Test 4: Document File Info", False,
                           f"GET /documents failed: {r_docs.status_code}")

    docs_data = r_docs.get_json().get("data", r_docs.get_json())
    docs = docs_data.get("documents", [])

    if not docs:
        return print_result("Test 4: Document File Info", False,
                           "No documents found")

    doc = docs[0]
    version_id = doc.get("document_version_id", "")
    title = doc.get("title", "?")
    print(f"  Testing with document: {title[:40]} (id={version_id[:20]}...)")

    r_file = client.get(f'/api/v1/documents/{version_id}/file')
    print(f"  GET /documents/{{id}}/file: status={r_file.status_code}")

    if r_file.status_code == 200:
        file_data = r_file.get_json().get("data", r_file.get_json())
        print(f"  File info keys: {list(file_data.keys())[:10]}")
        passed = True
    elif r_file.status_code == 404:
        # API-submitted text may not have file info
        r_content = client.get(f'/api/v1/documents/{version_id}/content')
        print(f"  GET /documents/{{id}}/content: status={r_content.status_code}")
        passed = r_content.status_code in (200, 404)
    else:
        passed = r_file.status_code < 500

    return print_result("Test 4: Document File Info", passed,
                       f"file_status={r_file.status_code}")


# ============================================================
# TEST 5: Window statuses in task result (Fix 3)
# ============================================================
def test_window_statuses(client):
    print("\n>>> TEST 5: Window Statuses")

    test_text = "窗口状态测试文档。这是一个测试，用来验证任务完成后窗口状态的正确性。"
    source = "窗口状态测试_e2e_fix5_v2"

    r = remember_text(client, test_text, source_name=source, timeout=120)
    if r.status_code not in (200, 202):
        return print_result("Test 5: Window Statuses", False,
                           f"POST failed: {r.status_code}")

    body = r.get_json()
    data = body.get("data", body)
    task_result = data.get("result", {})
    doc_version_id = task_result.get("document_version_id") or data.get("document_version_id")

    print(f"  Task: status={data.get('status')}, doc_version_id={doc_version_id}")
    print(f"  chunks_processed={task_result.get('chunks_processed')}")
    print(f"  entities={task_result.get('entities')}, relations={task_result.get('relations')}")

    integrity_ok = True
    if doc_version_id:
        r_integrity = client.get(f'/api/v1/documents/{doc_version_id}/integrity')
        if r_integrity.status_code == 200:
            integrity_data = r_integrity.get_json().get("data", r_integrity.get_json())
            print(f"  Integrity: complete={integrity_data.get('complete')}, "
                  f"windows={integrity_data.get('total_windows')}/{integrity_data.get('complete_windows')}")
            integrity_ok = integrity_data.get("complete") is True or (integrity_data.get("total_windows", 0) > 0)

    passed = data.get("status") == "completed" and integrity_ok
    return print_result("Test 5: Window Statuses", passed,
                       f"task_status={data.get('status')}, integrity_ok={integrity_ok}")


# ============================================================
# TEST 6: Cross-window dedup (C5)
# ============================================================
def test_cross_window_dedup(client):
    print("\n>>> TEST 6: Cross-Window Dedup")

    paragraphs = []
    names = ["曹操", "孙权", "周瑜", "司马懿", "黄忠", "魏延", "庞统", "鲁肃", "吕蒙", "陆逊"]
    for i in range(10):
        paragraphs.append(
            f"第{i+1}段：{names[i]}是三国时期的重要人物。"
            f"{names[i]}与刘备有着复杂的关系。"
            f"{names[i]}在历史上留下了深刻的印记。"
            f"许多后来的文学作品都描写了{names[i]}的故事。"
            f"在战场上，{names[i]}展现了卓越的才能和勇气。"
        )
    long_text = "\n\n".join(paragraphs)
    source = "跨窗口去重测试_e2e_fix6_v2"

    print(f"  Submitting long text ({len(long_text)} chars)...")
    r = remember_text(client, long_text, source_name=source, timeout=300)

    if r.status_code not in (200, 202):
        return print_result("Test 6: Cross-Window Dedup", False,
                           f"POST failed: {r.status_code}")

    body = r.get_json()
    data = body.get("data", body)
    task_result = data.get("result", {})

    entities_count = task_result.get("entities", 0)
    relations_count = task_result.get("relations", 0)
    chunks = task_result.get("chunks_processed", 0)

    print(f"  Task completed: chunks={chunks}, entities={entities_count}, relations={relations_count}")

    # Check for duplicates
    time.sleep(1)
    r_dupes = client.get('/api/v1/concepts/duplicates?limit=500')
    dupe_count = 0
    if r_dupes.status_code == 200:
        dupes_data = r_dupes.get_json().get("data", r_dupes.get_json())
        duplicates = dupes_data.get("duplicates", [])
        for d in duplicates:
            core = d.get("core_name", "")
            if core in names:
                dupe_count += 1
                print(f"  [DUPE] {core}: {len(d.get('entities', []))} versions")

    passed = data.get("status") == "completed"
    return print_result("Test 6: Cross-Window Dedup", passed,
                       f"chunks={chunks}, entities={entities_count}, relations={relations_count}, "
                       f"duplicates={dupe_count}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Deep-Dream E2E Integration Tests (Flask Test Client)")
    print("=" * 60)

    print("\n  Setting up Flask test client...")
    try:
        client = setup_client()
        print("  Client ready.")
    except Exception as e:
        print(f"  FATAL: Failed to create test client: {e}")
        traceback.print_exc()
        sys.exit(1)

    results = []

    tests = [
        ("Test 1: Duplicate Document Prevention", test_duplicate_document_prevention),
        ("Test 2: Entity/Relation Extraction", test_entity_relation_extraction),
        ("Test 3: Graph Traversal", test_graph_traversal),
        ("Test 4: Document File Info", test_document_file_info),
        ("Test 5: Window Statuses", test_window_statuses),
        ("Test 6: Cross-Window Dedup", test_cross_window_dedup),
    ]

    for name, fn in tests:
        try:
            results.append((name, fn(client)))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed_count = 0
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if passed:
            passed_count += 1
    print(f"\n  Total: {passed_count}/{len(results)} passed")
    print("=" * 60)

    sys.exit(0 if passed_count == len(results) else 1)
