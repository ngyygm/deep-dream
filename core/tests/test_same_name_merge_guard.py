"""P2.5 同名实体合并 embedding 守卫测试。

_resolve_same_name_conflicts 不再无条件合并同名 family：
- 双方都有已存 embedding 且余弦相似度 >= 0.75 → 合并（现状行为）
- 相似度 < 0.75 → 跳过合并并记 ambiguous（同名不同概念，如"苹果"公司 vs"苹果"水果）
- 任一方无 embedding 可比 → 维持合并（no-embedding-guard）

纯本地判断：守卫只读已持久化向量，不触发任何 LLM/embedding 编码调用
（_FakeEmbeddingClient 故意不实现 encode，一旦有人调用会立即暴露）。
"""
from datetime import datetime, timezone

import numpy as np

from core.models import Entity
from core.remember.alignment import _ResolutionMixin
from core.storage.sqlite.library_manager import LibraryManager


class _FakeEmbeddingClient:
    """仅声明可用——守卫只读已存向量；故意不实现 encode。"""

    model_name = "test-embedding"

    def is_available(self):
        return True


class _ResolverHarness(_ResolutionMixin):
    """最小宿主：只提供 _resolve_same_name_conflicts 依赖的 storage。"""

    def __init__(self, storage):
        self.storage = storage


def _vec_bytes(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()


def _save(mgr, fid, abs_id, name, content, emb=None):
    now = datetime.now(timezone.utc)
    mgr.save_entity(Entity(
        absolute_id=abs_id,
        family_id=fid,
        name=name,
        content=content,
        event_time=now,
        processed_time=now,
        # episode 不存在 → 落库为 NULL → 同 family 重复保存不去重（用于叠版本数）
        episode_id=f"ep_missing_{abs_id}",
        source_document="same_name_guard_test.md",
        embedding=emb,
    ))


def _make_mgr(tmp_path, with_embedding_client=False):
    mgr = LibraryManager(str(tmp_path / "lib"))
    if with_embedding_client:
        mgr.embedding_client = _FakeEmbeddingClient()
    return mgr


def test_high_similarity_same_name_merges(tmp_path):
    """同名 + 高相似 → 维持合并，重定向到版本数多的主实体。"""
    mgr = _make_mgr(tmp_path, with_embedding_client=True)
    _save(mgr, "fam_primary", "p_v1", "苹果", "一家消费电子公司",
          emb=_vec_bytes([1.0, 0.0, 1.0, 0.0]))
    _save(mgr, "fam_primary", "p_v2", "苹果", "一家消费电子公司",
          emb=_vec_bytes([1.0, 0.0, 1.0, 0.0]))
    _save(mgr, "fam_other", "o_v1", "苹果", "一家消费电子公司（重复抽取）",
          emb=_vec_bytes([1.0, 0.05, 1.0, 0.0]))

    resolver = _ResolverHarness(mgr)
    name_to_id, ambiguous = resolver._resolve_same_name_conflicts(
        {"苹果": ["fam_other", "fam_primary"]})

    assert name_to_id.get("苹果") == "fam_primary"
    assert ambiguous == set()
    assert mgr.resolve_family_id("fam_other") == "fam_primary"
    mgr.close()


def test_low_similarity_same_name_skips_merge(tmp_path):
    """同名 + 低相似（同名不同概念）→ 跳过合并、记歧义、不建名称映射。"""
    mgr = _make_mgr(tmp_path, with_embedding_client=True)
    _save(mgr, "fam_company", "c_v1", "苹果", "科技公司，生产 iPhone",
          emb=_vec_bytes([1.0, 0.0, 0.0, 0.0]))
    _save(mgr, "fam_company", "c_v2", "苹果", "科技公司，生产 iPhone",
          emb=_vec_bytes([1.0, 0.0, 0.0, 0.0]))
    _save(mgr, "fam_fruit", "f_v1", "苹果", "一种水果，蔷薇科苹果属植物",
          emb=_vec_bytes([0.0, 1.0, 0.0, 0.0]))

    resolver = _ResolverHarness(mgr)
    name_to_id, ambiguous = resolver._resolve_same_name_conflicts(
        {"苹果": ["fam_fruit", "fam_company"]})

    assert "苹果" not in name_to_id  # 歧义名不建映射，关系解析按 ambiguous 机制跳过
    assert ambiguous == {"苹果"}
    assert mgr.resolve_family_id("fam_fruit") == "fam_fruit"  # 未注册重定向
    mgr.close()


def test_no_embedding_keeps_merge(tmp_path):
    """任一方无已存 embedding → 维持合并现状（no-embedding-guard）。"""
    mgr = _make_mgr(tmp_path)  # 无 embedding client → 向量不落库
    _save(mgr, "fam_a", "a_v1", "苹果", "科技公司")
    _save(mgr, "fam_a", "a_v2", "苹果", "科技公司")
    _save(mgr, "fam_b", "b_v1", "苹果", "一种水果")

    resolver = _ResolverHarness(mgr)
    name_to_id, ambiguous = resolver._resolve_same_name_conflicts(
        {"苹果": ["fam_b", "fam_a"]})

    assert name_to_id.get("苹果") == "fam_a"
    assert ambiguous == set()
    assert mgr.resolve_family_id("fam_b") == "fam_a"
    mgr.close()


def test_dimension_mismatch_keeps_merge_without_raising(tmp_path):
    """双方向量维度不一致（换 embedding 模型后新旧共存）→ 按不可比维持合并，不抛错。"""
    mgr = _make_mgr(tmp_path, with_embedding_client=True)
    _save(mgr, "fam_3d", "d_v1", "苹果", "科技公司", emb=_vec_bytes([1.0, 0.0, 0.0]))
    _save(mgr, "fam_3d", "d_v2", "苹果", "科技公司", emb=_vec_bytes([1.0, 0.0, 0.0]))
    _save(mgr, "fam_4d", "e_v1", "苹果", "一种水果", emb=_vec_bytes([0.0, 1.0, 0.0, 0.0]))

    resolver = _ResolverHarness(mgr)
    name_to_id, ambiguous = resolver._resolve_same_name_conflicts(
        {"苹果": ["fam_4d", "fam_3d"]})

    # 维度不匹配的计算失败按"不可比"处理——维持合并现状且不抛异常打穿步骤9
    assert name_to_id.get("苹果") == "fam_3d"
    assert ambiguous == set()
    assert mgr.resolve_family_id("fam_4d") == "fam_3d"
    mgr.close()


def test_mixed_group_partial_merge_marks_ambiguous(tmp_path):
    """三 family 混合组：高相似并入主实体，低相似保持独立且名称记歧义。"""
    mgr = _make_mgr(tmp_path, with_embedding_client=True)
    _save(mgr, "fam_p", "mp_v1", "苹果", "科技公司", emb=_vec_bytes([1.0, 0.0, 0.0, 0.0]))
    _save(mgr, "fam_p", "mp_v2", "苹果", "科技公司", emb=_vec_bytes([1.0, 0.0, 0.0, 0.0]))
    _save(mgr, "fam_dup", "md_v1", "苹果", "科技公司（重复）",
          emb=_vec_bytes([0.9, 0.1, 0.0, 0.0]))
    _save(mgr, "fam_fruit", "mf_v1", "苹果", "一种水果",
          emb=_vec_bytes([0.0, 1.0, 0.0, 0.0]))

    resolver = _ResolverHarness(mgr)
    name_to_id, ambiguous = resolver._resolve_same_name_conflicts(
        {"苹果": ["fam_fruit", "fam_dup", "fam_p"]})

    # 高相似的 fam_dup 仍并入主实体
    assert mgr.resolve_family_id("fam_dup") == "fam_p"
    # 低相似的 fam_fruit 保持独立，名称记歧义、不建映射
    assert mgr.resolve_family_id("fam_fruit") == "fam_fruit"
    assert "苹果" not in name_to_id
    assert ambiguous == {"苹果"}
    mgr.close()
