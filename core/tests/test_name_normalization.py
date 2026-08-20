"""实体名归一化统一语义测试（P2）。

历史上存在 4 处语义不一致的实现（窗口去重 / 候选匹配 / FamilyWriteGate /
关系端点解析各判各的）。P2 起全部委托 core.utils.entity_match_key——
本文件钉住统一后的语义：任何一条断言翻转即身份语义变更，需按审计
不变式 (d) 重跑并发 ingest 检查（见 test_judge_service.TestGateWiring）。
"""

import pytest

from core.utils import entity_match_key, entity_name_variants
from core.remember.helpers import _core_entity_name
from core.remember._shared import normalize_entity_name_for_matching
from core.judge.models import norm_name


CASES = [
    # input                unified key          variants (raw, core)
    ("张三",                "张三",               ("张三", "张三")),
    ("张三教授",            "张三",               ("张三教授", "张三")),
    ("张三（北京大学）",     "张三",               ("张三（北京大学）", "张三")),
    ("张三(北大)",          "张三",               ("张三(北大)", "张三")),
    ("IBM",                "ibm",               ("IBM", "IBM")),
    ("ibm",                "ibm",               ("ibm", "ibm")),
    ("  Alice   Bob ",     "alice bob",         ("Alice Bob", "Alice Bob")),
    ("李四博士教授",         "李四",               ("李四博士教授", "李四")),
    ("曹操（魏王）",         "曹操",               ("曹操（魏王）", "曹操")),
]


@pytest.mark.parametrize("raw,key,variants", CASES)
def test_unified_key_and_variants(raw, key, variants):
    assert entity_match_key(raw) == key
    assert entity_name_variants(raw) == variants


@pytest.mark.parametrize("raw,key,variants", CASES)
def test_all_four_entrypoints_agree(raw, key, variants):
    """窗口去重 / 候选匹配 / gate key / judge 归一——同一输入同一结论。"""
    assert _core_entity_name(raw) == key
    assert normalize_entity_name_for_matching(raw) == key
    assert norm_name(raw) == key


def test_title_paren_case_unify_across_layers():
    """统一前 gate 判 '张三教授'≠'张三' 而匹配器判相等——竞态漏网路径。"""
    assert norm_name("张三教授") == norm_name("张三")
    assert normalize_entity_name_for_matching("IBM") == \
        normalize_entity_name_for_matching("ibm")
    assert _core_entity_name("张三（北京大学）") == _core_entity_name("张三")


def test_stackoverflow_names_keep_identity():
    """不相关名称不得因归一碰撞。"""
    assert entity_match_key("深度学习") != entity_match_key("机器学习")
    assert entity_match_key("AWS") != entity_match_key("Azure")
    # 变体兜底：全剥空时保留原文
    assert entity_match_key("（全括号）") == "（全括号）"


def test_resolver_prefix_recall_and_norm_filter(tmp_path):
    """resolve_family_id_from_conn：变体精确 + 前缀召回 + 归一过滤。"""
    import sqlite3
    from core.storage.sqlite.schema_v15 import init_schema_v15
    from core.judge.models import resolve_family_id_from_conn

    conn = sqlite3.connect(str(tmp_path / "g.db"))
    init_schema_v15(conn)
    for fid, name in [("f1", "张三教授"), ("f2", "IBM Corp"),
                      ("f3", "alice（旧称）")]:
        conn.execute(
            "INSERT INTO entity_families "
            "(entity_family_id, canonical_name, created_at, updated_at) "
            "VALUES (?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (fid, name))

    # 前缀召回：查询核心名 "张三" 命中全名存储行 "张三教授"
    assert resolve_family_id_from_conn(conn, "张三") == "f1"
    # 变体精确：括号注记剥离后命中
    assert resolve_family_id_from_conn(conn, "张三（北大）") == "f1"
    # NOCASE + 前缀误报被归一过滤（"alice" LIKE 'alice%' 召回 f2/f3，仅 f3 归一相等）
    assert resolve_family_id_from_conn(conn, "Alice") == "f3"
    assert resolve_family_id_from_conn(conn, "ibm corp") == "f2"
    assert resolve_family_id_from_conn(conn, "王五") is None
    conn.close()


def test_storage_backed_gate_converges_name_variants(tmp_path):
    """LibraryManager.find_family_id_by_name（gate _default_resolve 腿）。

    此前 gate 通过 getattr 探测该方法但它不存在——存储解析从未生效。
    """
    from datetime import datetime, timezone
    from core.models import Entity
    from core.storage.sqlite.library_manager import LibraryManager
    from core.judge import FamilyWriteGate

    now = datetime.now(timezone.utc)
    mgr = LibraryManager(str(tmp_path / "lib"))
    mgr.save_entity(Entity(
        absolute_id="e1", family_id="fam_zhang", name="张三教授",
        content="人物。", event_time=now, processed_time=now,
        episode_id="ep1", source_document="a.md"))
    gate = FamilyWriteGate(storage=mgr)
    assert gate.resolve_name("张三") == "fam_zhang"
    assert gate.resolve_name("张三（北大）") == "fam_zhang"
