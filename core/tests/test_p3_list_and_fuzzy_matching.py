"""P3 效率修复回归：GET /documents 轻量化（P3.7）与模糊子串限窗（P3.10）。

全部纯本地（tmp_path / 无 LLM / 无网络）。
"""
import uuid

from core.storage.sqlite import repositories as repos
from core.text_chunking import find_text_evidence


def _insert_document(storage, title: str, text: str, *, now: str = "2026-08-20T00:00:00"):
    """向测试库写入一篇完整入库的文档（documents + versions + 内容文件）。"""
    did = f"doc_{uuid.uuid4().hex[:10]}"
    vid = f"docver_{uuid.uuid4().hex[:10]}"
    managed = f"documents/{did}.md"
    content_dir = storage.library_path / "documents"
    content_dir.mkdir(parents=True, exist_ok=True)
    (storage.library_path / managed).write_text(text, encoding="utf-8")
    conn = storage._conn()
    repos.documents.insert_document(conn, did, title, managed, created_at=now, updated_at=now)
    repos.documents.insert_document_version(
        conn, vid, did, content_hash=uuid.uuid4().hex, version_content_path=managed,
        title=title, char_count=len(text), byte_size=len(text.encode("utf-8")), processed_at=now,
    )
    repos.documents.update_current_version(conn, did, vid, updated_at=now)
    conn.commit()
    return did, vid


_DOC_TEXT = "# 标题\n\n" + "".join(
    f"实体 Alpha-{i} 与 Beta-{i} 的关系在窗口中被观测。滑动窗口保持语义连续。\n" for i in range(30)
)


class TestDocumentsListLightweight:
    """P3.7：列表页默认不做逐文档 integrity 全量评估。"""

    def test_default_response_has_episode_count_no_integrity(self, client, processor):
        storage = processor.storage
        _did, vid = _insert_document(storage, "轻量化测试文档", _DOC_TEXT)
        resp = client.get("/api/v1/documents?graph_id=test_graph")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        docs = [d for d in data["documents"] if d.get("document_version_id") == vid]
        assert docs, "新插入的文档应出现在列表里"
        doc = docs[0]
        # 轻量字段就位；integrity 默认不再随列表返回（memory 页按需拉取）
        assert "episode_count" in doc
        assert "integrity" not in doc
        assert doc["role"] == "document"

    def test_explicit_config_gate_restores_legacy_integrity(self, test_app, processor):
        storage = processor.storage
        _did, vid = _insert_document(storage, "门控测试文档", _DOC_TEXT)
        # 显式开启 auto_check_documents（用户 opt-in）→ 旧的全量评估行为保留
        test_app.config["config"] = {"runtime": {"integrity": {"auto_check_documents": True}}}
        client = test_app.test_client()
        resp = client.get("/api/v1/documents?graph_id=test_graph")
        assert resp.status_code == 200
        docs = [d for d in resp.get_json()["data"]["documents"] if d.get("document_version_id") == vid]
        assert docs and "integrity" in docs[0]
        integrity = docs[0]["integrity"]
        assert integrity["complete"] in (True, False)
        assert integrity["total_windows"] >= 1

    def test_per_document_integrity_endpoint_available(self, client, processor):
        """按需端点（原有）保持可用：Web UI 的“检查”按钮走这里。"""
        storage = processor.storage
        _did, vid = _insert_document(storage, "按需检查文档", _DOC_TEXT)
        resp = client.get(f"/api/v1/documents/{vid}/integrity?graph_id=test_graph")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["success"] is True
        assert payload["data"]["total_windows"] >= 1


class TestSimilarSubstringBudget:
    """P3.10：模糊子串枚举受预算/早停约束，典型输入结果不变。"""

    BODY = (
        "深度记忆系统在文档优先切片策略下运行，实体 Alpha 与关系在窗口中被观测。"
        "embedding 模型对上下文窗口敏感。The quick brown fox jumps over the lazy dog."
    ) * 4

    def test_fuzzy_match_still_found(self):
        # 与文本部分相似（非精确/归一化可命中）的名字仍要给出 similar_substring 证据
        needle = "深度记忆系统在文档优先切片策略下运算实体"
        hits = find_text_evidence(self.BODY, [needle], limit=3)
        assert hits, "限窗后模糊命中不应丢失"
        assert hits[0]["match_type"] == "similar_substring"
        assert hits[0]["confidence"] >= 0.78
        assert self.BODY[hits[0]["start_offset"]:hits[0]["end_offset"]]

    def test_no_match_stays_no_match(self):
        # 完全不相关的长名仍应无命中（预算截断不会凭空造出低质匹配）
        needle = "量子纠缠超导材料冶金工艺规程说明书附录三修订版全文"
        assert find_text_evidence(self.BODY, [needle], limit=1) == []

    def test_exact_and_normalized_paths_unaffected(self):
        hits = find_text_evidence(self.BODY, ["Alpha"], limit=1)
        assert hits and hits[0]["match_type"] in ("exact", "normalized")

    def test_long_needle_bounded_latency(self):
        """长实体名（原 O(n²) 全枚举）必须在有界时间内完成。"""
        import time
        needle = "这是一个完全不存在于文本之中的超长实体名称用于验证限窗之后的枚举成本有上界xyz"
        body = self.BODY * 3  # 加长正文提高旧实现的枚举量
        t0 = time.perf_counter()
        find_text_evidence(body, [needle], limit=1)
        elapsed = time.perf_counter() - t0
        # 旧实现同输入 >1s（n≈45、正文 ~1100 字符时全枚举近万窗口）；
        # 预算约束后应稳定在几十毫秒量级，这里放宽到 0.5s 防慢机器误报。
        assert elapsed < 0.5, f"限窗后仍耗时 {elapsed*1000:.0f} ms"

    def test_repetitive_long_needle_not_silently_dropped(self):
        """seq2 固定为 needle 后 autojunk 不得作用于 needle（回归验证）。

        背景：matcher 复用把 difflib 的 autojunk（仅作用于 seq2、序列 ≥200 且
        高频字符 >1% 时整类字符被剔除）从候选窗一侧挪到了 needle 一侧——
        n≥200 的名字会被整体 junk 成 0 匹配，旧版能命中（0.951）而新版返回空。
        关掉 autojunk 后应恢复命中；旧版对该输入的期望值为 0.951（差分实测）。
        """
        needle = (
            "Zf体文5f8义n排策略d先da0bb索度统0beg排a重0emgcd义2口6eg义4优口71deam窗b系5语索排7忆9档0"
            "d文记文0d5义g4n口先系口口语08d文6略0e3b1排b7实1窗dm度cbd8实e先d先深75系重4索策d窗先g6检"
            "1c08fn6gg深语1m序fg深de忆统策e4e度5系记7策6义3深口口序关义gY排87策排de系记系序系n窗体3a"
            "实b深e记1序7略d先记cd6文重忆文系深索排文实忆先eb深bed0X005X39e深ca优e忆3Y文fg系bnZd"
        )
        body = (
            "Zf体文5实8义n排策略d先da0bb索度统0beg排a重0emgcd义2口6eg义4优口71deam窗b系5c索排7忆9档0"
            "d文记文0d5义g4n口先系口口语08d文6略0e3b1排b7实1统dm度cbd8实e先d先深75系重4索策d窗先g6检"
            "1c08fn6gg深语1m序fg深de忆统策e4e度5系记7策6义3深口口序关义gY排87策系de系记系序系n窗体3a"
            "实b深b记1序7略d先记cd6文重忆文系深索排文实忆先eb深bed0X005X39e深ca优e忆bY文fg系bnZd。结尾。"
        )
        hits = find_text_evidence(body, [needle], limit=1)
        assert hits, "重复模式构成的超长名（n=240）不应因 autojunk 作用于 needle 而被静默丢弃"
        assert hits[0]["match_type"] == "similar_substring"
        assert hits[0]["confidence"] >= 0.78
