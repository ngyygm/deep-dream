"""P2 精准性回归：round-4 substring 收紧、兜底共现关系门控。

全部纯本地（stub storage / 无 LLM），不发真实请求。
"""
from types import SimpleNamespace

from core.remember.alignment import _ResolutionMixin
from core.remember.alignment_relations import _OrphanMixin


class _NameStorageStub:
    """只实现 _resolve_missing_relation_entity_names 需要的存储面。"""

    def get_family_ids_by_names(self, names):
        return {}


class _ResolverStub(_ResolutionMixin):
    storage = _NameStorageStub()


def _resolve(known_map, missing_via_relation):
    stub = _ResolverStub()
    pending = [{"entity1_name": n, "entity2_name": "锚点"} for n in missing_via_relation]
    known = dict(known_map)
    known.setdefault("锚点", "fid_anchor")
    name_to_id, db_matched, fuzzy = stub._resolve_missing_relation_entity_names(
        pending, known, ambiguous_duplicate_names=set())
    return name_to_id


class TestRound4SubstringTightening:
    """无 LLM 验证的 round-4 兜底：包含比 ≥0.5 才归并。"""

    def test_alias_within_ratio_resolves(self):
        m = _resolve({"甄士隐": "fid_a"}, ["士隐"])
        assert m["士隐"] == "fid_a"

    def test_boundary_ratio_resolves(self):
        # 2 字核 ⊂ 4 字名：short*2 == long，放行
        m = _resolve({"通灵宝玉": "fid_b"}, ["宝玉"])
        assert m["宝玉"] == "fid_b"

    def test_missing_longer_than_known_resolves(self):
        m = _resolve({"宝玉": "fid_c"}, ["通灵宝玉"])
        assert m["通灵宝玉"] == "fid_c"

    def test_coincidental_substring_rejected(self):
        # "公司"(2) ⊂ "阿里巴巴集团公司"(8)：包含比 0.25，拒绝
        m = _resolve({"阿里巴巴集团公司": "fid_d"}, ["公司"])
        assert "公司" not in m


class _OrphanStub(_OrphanMixin):
    """只实现 _cleanup_orphaned_entities 前半段需要的存储面。"""

    def __init__(self, enabled):
        self.remember_fallback_cooccurrence = enabled
        self.storage = SimpleNamespace(
            batch_get_entity_degrees=lambda fids: {"f_orphan": 0},
            get_entity_version_counts=lambda fids: {"f_orphan": 1},
        )
        self.creator_calls = 0

    def _create_fallback_cooccurrence_relations(self, orphan_fids, *args, **kwargs):
        self.creator_calls += 1
        self.last_orphans = list(orphan_fids)
        return len(orphan_fids)


_ORPHAN_ENTITIES = [SimpleNamespace(family_id="f_orphan", name="孤实体")]


class TestFallbackCooccurrenceGate:
    """兜底共现关系（confidence 0.3 轮询配对）默认关闭。"""

    def test_disabled_by_default(self):
        stub = _OrphanStub(enabled=False)
        stub._cleanup_orphaned_entities(_ORPHAN_ENTITIES, verbose=False)
        assert stub.creator_calls == 0

    def test_enabled_explicitly(self):
        stub = _OrphanStub(enabled=True)
        stub._cleanup_orphaned_entities(_ORPHAN_ENTITIES, verbose=False)
        assert stub.creator_calls == 1
        assert stub.last_orphans == ["f_orphan"]


class TestFallbackFlagPlumbing:
    def _resolved(self, remember_config):
        from core.remember.orchestrator import TemporalMemoryGraphProcessor

        class _Cfg:
            _resolve_remember_config = TemporalMemoryGraphProcessor._resolve_remember_config

        cfg = _Cfg()
        cfg._resolve_remember_config(
            None, remember_config, None, None, None, None, None, None)
        return cfg

    def test_default_off(self):
        assert self._resolved({}).remember_fallback_cooccurrence is False

    def test_explicit_on(self):
        cfg = self._resolved({"fallback_cooccurrence_relations": True})
        assert cfg.remember_fallback_cooccurrence is True
