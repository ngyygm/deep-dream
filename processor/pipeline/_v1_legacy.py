"""V1-only extraction functions (legacy, not used by V2/V3 pipelines).

Archived here for reference. Production code uses V2/V3 pipeline
(new_extraction.py). Tests that reference these functions import from
this module.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .extraction_utils import _clean_entity_name, MIN_RELATION_CONTENT_LENGTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_pair_for_relation(e1: str, e2: str) -> Tuple[str, str]:
    """无向边端点按字典序固定。委托给 processor.utils.normalize_entity_pair。"""
    from processor.utils import normalize_entity_pair
    return normalize_entity_pair(e1, e2)


# ---------------------------------------------------------------------------
# V1 entity name validation constants
# ---------------------------------------------------------------------------

# 非实体文本片段黑名单
_NON_ENTITY_FRAGMENTS: set = {
    "遂得", "险恶", "处理进度", "突阵而出", "诸营已失", "弃关",
    "相拒", "火来", "营已失", "失了", "遂", "险", "恶",
    "忠心耿耿的群体", "官職", "居住地",
}

_MAX_ENTITY_NAME_LENGTH = 30
_MIN_ENTITY_NAME_LENGTH = 2


# ---------------------------------------------------------------------------
# V1 entity name validation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _is_valid_entity_name(name: str) -> bool:
    """检查实体名称是否是有效的概念实体名称（而非文本片段或无意义词）。"""

    if not name or len(name) < _MIN_ENTITY_NAME_LENGTH:
        return False

    if len(name) == 2 and re.match(r'^[\u4e00-\u9fff]+$', name):
        _verb_2char = {
            "主持", "负责", "建议", "赞同", "大悦", "修缮",
            "广纳", "满意", "感慨", "内斗", "恢复", "统一",
            "开发", "设计", "建立", "实现", "管理", "处理",
            "分析", "评估", "研究", "探讨", "优化", "改进",
            "迁移", "发现", "阅读", "合作", "超越",
        }
        if name in _verb_2char:
            return False

    _has_en_paren = bool(re.search(r'[(（][A-Za-z]', name))
    _max_len = 50 if (name.isascii() and ' ' in name) or _has_en_paren else _MAX_ENTITY_NAME_LENGTH
    if len(name) > _max_len:
        return False

    if name in _NON_ENTITY_FRAGMENTS:
        return False

    _system_patterns_cn = [
        "处理进度", "系统状态", "步骤", "缓存", "抽取", "对齐",
        "窗口", "流水线", "阈值", "配置", "待处理", "已完成",
    ]
    for pat in _system_patterns_cn:
        if pat in name:
            return False
    _system_patterns_en = [
        (r'\bToken\b', re.IGNORECASE),
        (r'\bAPI\b', re.IGNORECASE),
        (r'\bLLM\b', re.IGNORECASE),
        (r'\bJSON\b', re.IGNORECASE),
    ]
    for pat, flags in _system_patterns_en:
        if re.search(pat, name, flags):
            _stripped = re.sub(r'[（(][^）)]+[)）]', '', name).strip()
            if _stripped and re.match(r'^[A-Za-z][A-Za-z0-9_.+\-/]+$', _stripped):
                continue
            return False

    for i in range(len(name) - 5):
        fragment = name[i:i+2]
        if len(fragment) == 2 and fragment == name[i+2:i+4] == name[i+4:i+6]:
            return False

    _dialogue_markers = ["曰：", "道：", "笑曰", "怒曰", "叹曰", "言曰", "喝曰",
                         "答曰", "问曰", "叫曰", "说道", "说道：",
                         "哀哉", "痛哉", "惜哉", "壮哉"]
    for marker in _dialogue_markers:
        if marker in name:
            return False

    if "！" in name and len(name) > 4:
        return False

    if name.isascii() and len(name) > 6:
        if name.endswith('!') or name.endswith('?'):
            return False

    if "、" in name:
        return False

    if name.isascii() and name.count(',') >= 3:
        return False

    if "，" in name and len(name) > 8:
        return False

    if name.isascii() and '. ' in name:
        return False

    _core_for_de = re.sub(r'[（(][^）)]+[）)]', '', name).strip()
    if "的" in _core_for_de and len(_core_for_de) > 6:
        return False

    if '_' in name and re.match(r'^[a-z][a-z0-9_]+$', name):
        return False

    paren_match = re.findall(r'（([^）]+)）', name)
    for paren_content in paren_match:
        if any(c in paren_content for c in "，。！？、；："):
            return False
        for pat in _system_patterns_cn:
            if pat in paren_content:
                return False

    _year_match = re.match(r'^(\d{4}年?(?:\d{1,2}月?(?:\d{1,2}日?)?)?)', name)
    if _year_match:
        _year_part = _year_match.group(1)
        _after_year = name[len(_year_part):].strip()
        _after_clean = re.sub(r'[（(][^）)]+[)）]', '', _after_year).strip()
        if not _after_clean:
            return False

    core_chinese = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
    core_chinese = re.sub(r'（[A-Za-z\s]+）$', '', core_chinese).strip()

    _too_generic = {
        "投资", "研究", "开发", "分析", "比较", "测试", "设计",
        "管理", "处理", "调整", "优化", "改进", "建立", "实现",
        "投资方", "研发", "探讨", "评估",
        "迁移", "合作研究", "开创性研究", "元素发现",
        "计算任务", "特定计算任务", "叠加态", "计算能力",
        "超越", "投资行为", "重要成就",
        "开源", "文档名", "文件名",
        "知名项目", "开源方式", "开源方式发布",
        "设计理念", "设计目标", "设计发布",
        "编程效率", "提高编程效率",
        "静态强类型", "编译型语言",
        "网络服务器开发",
        "放射性研究", "科学历史", "科学遗产",
        "主持", "负责", "建议", "赞同", "遂令", "大悦",
        "修缮", "广纳", "修缮城防", "广纳贤才",
        "大业", "贤才", "百姓", "民心", "民生",
        "内政", "军事防务", "国事", "城防",
        "满意", "农业", "开源语言",
        "科学成就", "科学贡献", "科学传承", "科学史", "科学合作",
        "科学影响", "科学教育", "科学界", "研究方法",
        "学术生涯", "科学界地位",
        "科学成就传播", "科学成就历史意义", "科学成就国际影响",
        "科学成就影响", "科学成就社会影响", "科学成就认可",
        "科学贡献评价",
        "发现", "阅读", "合作", "丈夫", "妻子",
        "唯一", "开创性", "第一位", "元素",
        "悲痛", "战败", "病逝", "病逝时间",
        "社会秩序", "继承权争夺", "统一北方",
        "军事和政治策略", "谋士",
        "军事行动", "悲痛之情", "谋士角色",
        "屯田制度实施",
        "可移植", "可移植性", "大规模", "自动化", "自动部署",
        "扩展", "伸缩", "可伸缩", "可伸缩性",
        "资源使用", "资源管理", "资源优化",
        "容器化", "容器化应用", "容器集群",
        "多容器", "多容器应用",
    }
    if name in _too_generic or core_chinese in _too_generic:
        return False

    _meta_names = frozenset({
        "章节标题", "回目标题", "节标题", "小节名", "小节标题",
        "列表主题", "段落主题", "章节主题", "节名",
        "文档名", "文件名", "文档标题", "文件标题",
        "当前位置", "当前位置标识",
    })
    if name in _meta_names:
        return False

    if re.match(r'^(文件名|文档名|文档|文件|标题)[：:]', name):
        return False

    if name.endswith("角色") and len(name) <= 6:
        return False

    if name.endswith("之情") and len(name) <= 6:
        return False

    _english_generic = {
        "research", "discovery", "study", "analysis", "method",
        "partnership", "collaboration", "husband", "wife", "read",
        "influence", "impact", "education", "contribution",
        "achievement", "legacy", "history", "status", "role",
        "first", "pioneering", "only", "one and only", "elements",
        "development", "improvement", "enhancement", "optimization",
        "integration", "implementation", "management", "migration",
        "processing", "handling", "testing", "review", "feedback",
        "quality", "performance", "reliability", "efficiency",
        "scalability", "availability", "security", "stability",
        "importance", "significance", "relevance",
        "process", "procedure", "approach", "strategy", "solution",
        "framework", "architecture", "structure", "design", "pattern",
        "practice", "principle", "concept", "idea", "overview",
        "introduction", "background", "summary", "conclusion",
        "new elements", "pioneering research", "scientific contribution",
        "academic studies", "discovery of new elements",
        "scientific history", "scientific heritage", "scientific legacy",
        "radioactive research", "medical assistance", "women scientists",
        "best practice", "key factor", "main feature", "core concept",
        "important aspect", "significant development", "major change",
        "recent advance", "current status", "future direction",
        "relationship", "connection", "association", "difference",
        "similarity", "comparison", "advantage", "disadvantage",
        "benefit", "challenge", "issue", "problem", "solution",
    }
    _en_paren = re.findall(r'[(（]([A-Za-z][A-Za-z\s]+)[)）]', name)
    for en in _en_paren:
        if en.strip().lower() in _english_generic:
            return False
    if name.isascii() and name.lower().strip() in _english_generic:
        return False

    if name.isascii():
        _en_generic_start = ("pioneering ", "discovery of ", "study of ",
                             "analysis of ", "history of ", "impact of ",
                             "overview of ", "introduction to ", "review of ",
                             "basics of ", "fundamentals of ", "principles of ",
                             "guide to ", "approach to ", "step in ",
                             "role of ", "importance of ", "future of ",
                             "state of ", "evolution of ", "development of ")
        _en_generic_end = (" research", " discovery", " contribution",
                           " achievement", " method", " study",
                           " analysis", " collaboration", " partnership",
                           " education", " influence", " impact",
                           " legacy", " history", " studies",
                           " overview", " introduction", " summary",
                           " conclusion", " review", " perspective",
                           " development", " improvement", " enhancement",
                           " optimization", " integration", " management",
                           " framework", " architecture", " design")
        name_lower = name.lower().strip()
        for start in _en_generic_start:
            if name_lower.startswith(start):
                return False
        for end in _en_generic_end:
            if name_lower.endswith(end):
                return False

    if name.isascii():
        _en_dialogue_markers = ['said "', 'said that', 'says that', 'stated that',
                                'according to', 'claims that', 'argues that',
                                'believes that', 'mentioned that', 'noted that']
        name_lower = name.lower()
        for marker in _en_dialogue_markers:
            if name_lower.startswith(marker):
                return False

    if name.isascii() and ' ' in name:
        _en_verb_start_patterns = [
            r'^(?:improving|enhancing|optimizing|managing|handling|processing|building|creating|developing|designing|implementing|testing|deploying|monitoring|analyzing|evaluating)\b',
            r'^(?:how to|why|when|where|what|which)\b',
            r'^(?:the|a|an)\s+(?:best|most|main|key|important|significant|major|new|latest|current|future)\b',
        ]
        name_lower = name.lower().strip()
        for pat in _en_verb_start_patterns:
            if re.match(pat, name_lower):
                return False

    _generic_prefixes = [
        "科学成就", "科学贡献", "科学界", "学术", "研究",
        "设计",
    ]
    for prefix in _generic_prefixes:
        if name.startswith(prefix) and len(name) > len(prefix) and len(name) <= 8:
            suffix = name[len(prefix):]
            _generic_suffixes = {
                "传播", "历史意义", "国际影响", "影响", "社会影响",
                "认可", "评价", "地位", "发展", "意义", "价值",
                "概述", "总结", "背景", "特点", "方法",
                "理念", "目标", "发布", "思想", "方向", "趋势",
            }
            if suffix in _generic_suffixes:
                return False

    _action_suffixes = ["研究", "发现", "探索", "分析", "评价", "讨论", "传播"]
    _check_name = core_chinese or name
    if len(_check_name) >= 4 and len(_check_name) <= 6:
        for suffix in _action_suffixes:
            if _check_name.endswith(suffix):
                prefix = _check_name[:-len(suffix)]
                _generic_action_prefixes = {
                    "元素", "开创性", "合作", "系统", "深入", "初步",
                    "全面", "综合", "整体", "长期", "持续", "新",
                }
                if prefix in _generic_action_prefixes:
                    return False

    _verb_phrase_patterns = [
        r'^(恢复|稳定|统一|建立|推行|实施|开展|加强|推进|促进|巩固|维护|保障|完成|实现|争取|争夺|继承权)',
        r'^(提高|提升|增强|改善|优化|加速)',
        r'^自动(化|部署|扩展|伸缩|管理|配置|监控)',
        r'^(感慨|内斗)',
        r'病逝$',
        r'去世$',
        r'争斗$',
        r'的话语$',
    ]
    for pat in _verb_phrase_patterns:
        if re.search(pat, core_chinese or name):
            return False

    return True


# ---------------------------------------------------------------------------
# V1 entity dedup
# ---------------------------------------------------------------------------

def _core_entity_name(name: str) -> str:
    """提取实体名称的核心部分（去掉所有括号），用于去重比较。"""
    return re.sub(r'[（(][^）)]+[）)]', '', name).strip()


def dedupe_extracted_entities(entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """按实体 name（strip 后）去重；同名时保留 content 更长的条目。同时过滤非实体文本片段。"""
    name_to_index: Dict[str, int] = {}
    core_name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue

        cleaned_name = _clean_entity_name(name)
        if cleaned_name != name:
            name = cleaned_name

        if not _is_valid_entity_name(name):
            continue

        content = str(e.get("content") or "").strip()
        if len(content) < 8:
            continue

        existing_idx = name_to_index.get(name)
        if existing_idx is not None:
            if len(content) > len(out[existing_idx]["content"]):
                out[existing_idx] = {"name": name, "content": content}
            continue

        core_name = _core_entity_name(name)
        core_idx = core_name_to_index.get(core_name)
        if core_idx is not None:
            existing = out[core_idx]
            if len(name) > len(existing["name"]):
                existing["name"] = name
            name_to_index[name] = core_idx
            if len(content) > len(existing["content"]):
                existing["content"] = content
            continue

        idx = len(out)
        name_to_index[name] = idx
        if core_name != name:
            core_name_to_index[core_name] = idx
        out.append({"name": name, "content": content})
    return out


# ---------------------------------------------------------------------------
# V1 recalled entity handling
# ---------------------------------------------------------------------------

_RECALL_ENTITY_SYSTEM_PATTERNS = (
    "处理进度", "系统状态", "步骤", "缓存", "抽取", "窗口", "流水线",
    "token", "api", "json", "llm",
)


def _is_viable_recalled_entity_name(name: str) -> bool:
    """概念候选阶段的轻量名称校验。"""
    if not name:
        return False
    if len(name) < _MIN_ENTITY_NAME_LENGTH:
        return False
    if len(name) > 60:
        return False
    if name in _NON_ENTITY_FRAGMENTS:
        return False
    lowered = name.lower()
    if any(pattern in lowered for pattern in _RECALL_ENTITY_SYSTEM_PATTERNS):
        return False
    if re.match(r'^(文件名|文档名|文档|文件|标题)[：:]', name):
        return False
    if "曰：" in name or "道：" in name or "说道" in name:
        return False
    if "、" in name:
        return False
    if name.isascii() and (name.endswith('!') or name.endswith('?')):
        return False
    return True


def dedupe_recalled_entities(entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Remember 多步骤召回阶段的轻量去重。"""
    name_to_index: Dict[str, int] = {}
    core_name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        name = _clean_entity_name(str(entity.get("name") or "").strip())
        if not _is_viable_recalled_entity_name(name):
            continue
        content = str(entity.get("content") or "").strip()
        candidate = {"name": name, "content": content}

        existing_idx = name_to_index.get(name)
        if existing_idx is not None:
            existing = out[existing_idx]
            if len(candidate["content"]) > len(existing.get("content", "")):
                out[existing_idx] = candidate
            continue

        core_name = _core_entity_name(name)
        core_idx = core_name_to_index.get(core_name)
        if core_idx is not None:
            existing = out[core_idx]
            if len(name) > len(existing["name"]):
                existing["name"] = name
            if len(candidate["content"]) > len(existing.get("content", "")):
                existing["content"] = candidate["content"]
            name_to_index[name] = core_idx
            continue

        idx = len(out)
        out.append(candidate)
        name_to_index[name] = idx
        core_name_to_index.setdefault(core_name, idx)
    return out


def stabilize_recalled_entities_with_report(
    entities: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Remember 概念召回阶段的稳定化版本，返回保留结果与被拒原因。"""
    name_to_index: Dict[str, int] = {}
    core_name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    for entity in entities or []:
        if not isinstance(entity, dict):
            rejected.append({"reason": "invalid_payload", "name": "", "cleaned_name": ""})
            continue
        raw_name = str(entity.get("name") or "").strip()
        cleaned_name = _clean_entity_name(raw_name)
        if not _is_viable_recalled_entity_name(cleaned_name):
            rejected.append({
                "reason": "invalid_name",
                "name": raw_name,
                "cleaned_name": cleaned_name,
            })
            continue
        content = str(entity.get("content") or "").strip()
        candidate = {"name": cleaned_name, "content": content}

        existing_idx = name_to_index.get(cleaned_name)
        if existing_idx is not None:
            existing = out[existing_idx]
            if len(candidate["content"]) > len(existing.get("content", "")):
                out[existing_idx] = candidate
                rejected.append({
                    "reason": "duplicate_name_replaced_shorter_content",
                    "name": raw_name,
                    "cleaned_name": cleaned_name,
                    "kept_name": cleaned_name,
                })
            else:
                rejected.append({
                    "reason": "duplicate_name",
                    "name": raw_name,
                    "cleaned_name": cleaned_name,
                    "kept_name": cleaned_name,
                })
            continue

        core_name = _core_entity_name(cleaned_name)
        core_idx = core_name_to_index.get(core_name)
        if core_idx is not None:
            existing = out[core_idx]
            kept_name = existing["name"]
            if len(cleaned_name) > len(existing["name"]):
                existing["name"] = cleaned_name
                kept_name = cleaned_name
            if len(candidate["content"]) > len(existing.get("content", "")):
                existing["content"] = candidate["content"]
            name_to_index[cleaned_name] = core_idx
            rejected.append({
                "reason": "duplicate_core_name",
                "name": raw_name,
                "cleaned_name": cleaned_name,
                "kept_name": kept_name,
            })
            continue

        idx = len(out)
        out.append(candidate)
        name_to_index[cleaned_name] = idx
        core_name_to_index.setdefault(core_name, idx)

    return out, rejected


# ---------------------------------------------------------------------------
# V1 relation dedup
# ---------------------------------------------------------------------------

def dedupe_relation_candidates(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """关系候选去重：按无向实体对保留最长线索。"""
    pair_to_index: Dict[Tuple[str, str], int] = {}
    out: List[Dict[str, str]] = []
    for relation in relations or []:
        if not isinstance(relation, dict):
            continue
        entity1_name = str(relation.get("entity1_name") or relation.get("from_entity_name") or "").strip()
        entity2_name = str(relation.get("entity2_name") or relation.get("to_entity_name") or "").strip()
        if not entity1_name or not entity2_name or entity1_name == entity2_name:
            continue
        pair = _normalize_pair_for_relation(entity1_name, entity2_name)
        hint = str(relation.get("content") or "").strip()
        existing_idx = pair_to_index.get(pair)
        candidate = {
            "entity1_name": pair[0],
            "entity2_name": pair[1],
            "content": hint,
        }
        if existing_idx is None:
            pair_to_index[pair] = len(out)
            out.append(candidate)
            continue
        if len(hint) > len(out[existing_idx].get("content", "")):
            out[existing_idx]["content"] = hint
    return out


def normalize_relation_candidates_with_report(
    relations: Optional[List[Dict[str, Any]]],
    valid_entity_names: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """关系候选标准化，返回去重结果与拒绝原因。"""
    pair_to_index: Dict[Tuple[str, str], int] = {}
    out: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    _valid_lower: Dict[str, str] = {}
    _valid_core: Dict[str, str] = {}
    if valid_entity_names:
        for name in valid_entity_names:
            _valid_lower[name.lower()] = name
            core = _core_entity_name(name)
            if core and core not in _valid_core:
                _valid_core[core] = name

    def _resolve_to_valid(raw_name: str) -> Optional[str]:
        if not valid_entity_names:
            return raw_name
        if raw_name in valid_entity_names:
            return raw_name
        low = raw_name.lower()
        if low in _valid_lower:
            return _valid_lower[low]
        core = _core_entity_name(raw_name)
        if core in _valid_core:
            return _valid_core[core]
        if core and len(core) >= 2:
            for vname in valid_entity_names:
                vcore = _core_entity_name(vname)
                if vcore and len(vcore) >= 2:
                    if core.lower() in vcore.lower() or vcore.lower() in core.lower():
                        return vname
        return None

    for relation in relations or []:
        if not isinstance(relation, dict):
            rejected.append({"reason": "invalid_payload"})
            continue
        entity1_name = str(relation.get("entity1_name") or relation.get("from_entity_name") or "").strip()
        entity2_name = str(relation.get("entity2_name") or relation.get("to_entity_name") or "").strip()
        hint = str(relation.get("content") or "").strip()
        if not entity1_name or not entity2_name:
            rejected.append({
                "reason": "missing_endpoint",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        if entity1_name == entity2_name:
            rejected.append({
                "reason": "self_relation",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        pair = _normalize_pair_for_relation(entity1_name, entity2_name)
        if valid_entity_names:
            resolved_a = _resolve_to_valid(pair[0])
            resolved_b = _resolve_to_valid(pair[1])
            if not resolved_a or not resolved_b:
                rejected.append({
                    "reason": "unknown_endpoint",
                    "entity1_name": pair[0],
                    "entity2_name": pair[1],
                })
                continue
            pair = _normalize_pair_for_relation(resolved_a, resolved_b)
        candidate = {
            "entity1_name": pair[0],
            "entity2_name": pair[1],
            "content": hint,
        }
        existing_idx = pair_to_index.get(pair)
        if existing_idx is None:
            pair_to_index[pair] = len(out)
            out.append(candidate)
            continue
        if len(hint) > len(out[existing_idx].get("content", "")):
            out[existing_idx]["content"] = hint
            rejected.append({
                "reason": "duplicate_pair_replaced_shorter_hint",
                "entity1_name": pair[0],
                "entity2_name": pair[1],
            })
        else:
            rejected.append({
                "reason": "duplicate_pair",
                "entity1_name": pair[0],
                "entity2_name": pair[1],
            })

    return out, rejected


# ---------------------------------------------------------------------------
# V1 helpers
# ---------------------------------------------------------------------------

def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _is_valid_relation_content(content: str, entity1_name: str = "", entity2_name: str = "") -> bool:
    """检查关系内容是否是有效的关系描述。"""
    if not content or len(content) < MIN_RELATION_CONTENT_LENGTH:
        return False

    _label_only_patterns = {
        "官職", "居住地", "主从关系", "同僚关系", "敌对关系",
        "合作关系", "亲属关系", "上下级关系", "竞争关系",
        "朋友关系", "师徒关系", "同事关系", "邻居关系",
        "所属关系", "从属关系", "包含关系", "依赖关系",
        "colleague", "colleagues", "partnership", "collaboration",
        "competitor", "competitors", "neighbor", "neighbors",
        "relationship", "connection", "association",
    }
    if content.strip() in _label_only_patterns or content.strip().lower() in _label_only_patterns:
        return False

    _empty_patterns = [
        r'^.{0,4}与.{0,4}的?关联关系?$',
        r'^.{0,4}与.{0,4}有关$',
        r'^.{0,4}和.{0,4}的关系?$',
        r'^.{0,4}与.{0,4}存在关联$',
        r'^.{0,4}和.{0,4}相关$',
        r'^.{0,30}\bis\s+related\s+to\s+.{0,30}$',
        r'^.{0,30}\bis\s+associated\s+with\s+.{0,30}$',
        r'^.{0,30}\bis\s+connected\s+to\s+.{0,30}$',
        r'^.{0,30}\bis\s+linked\s+to\s+.{0,30}$',
        r'^.{0,30}\band\s+.{0,30}\s+have\s+a\s+relationship$',
        r'^.{0,30}\band\s+.{0,30}\s+are\s+related$',
        r'^.{0,30}\band\s+.{0,30}\s+are\s+associated$',
        r'^.{0,30}\band\s+.{0,30}\s+are\s+connected$',
        r'^has\s+a\s+relationship\s+with\s+.{0,30}$',
        r'^there\s+is\s+a\s+(?:relationship|connection|association)\s+between\b.{0,50}$',
    ]
    for pattern in _empty_patterns:
        if re.match(pattern, content.strip(), re.IGNORECASE):
            return False

    return True


def validate_written_entities_with_report(
    entities: Optional[List[Dict[str, Any]]],
    stable_names: List[str],
    fallback_content_by_name: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """校验实体写作结果，确保实体名稳定且 content 不为空。"""
    stable_name_set = set(stable_names)
    fallback_content_by_name = fallback_content_by_name or {}
    name_to_best: Dict[str, Dict[str, str]] = {}
    rejected: List[Dict[str, str]] = []

    for entity in entities or []:
        if not isinstance(entity, dict):
            rejected.append({"reason": "invalid_payload", "name": ""})
            continue
        name = _clean_entity_name(str(entity.get("name") or "").strip())
        if not name:
            rejected.append({"reason": "missing_name", "name": ""})
            continue
        if stable_name_set and name not in stable_name_set:
            rejected.append({"reason": "unknown_stable_name", "name": name})
            continue
        content = str(entity.get("content") or "").strip()
        if not content:
            content = fallback_content_by_name.get(name, "").strip()
        if not content:
            content = f'文本中提到了\u201c{name}\u201d这一概念。'
            rejected.append({"reason": "empty_content_fallback", "name": name})

        candidate = {"name": name, "content": content}
        existing = name_to_best.get(name)
        if existing is None:
            name_to_best[name] = candidate
            continue
        if len(content) > len(existing.get("content", "")):
            name_to_best[name] = candidate
            rejected.append({"reason": "duplicate_name_replaced_shorter_content", "name": name})
        else:
            rejected.append({"reason": "duplicate_name", "name": name})

    out: List[Dict[str, str]] = []
    for name in stable_names:
        candidate = name_to_best.get(name)
        if candidate is not None:
            out.append(candidate)
            continue
        fallback = fallback_content_by_name.get(name, "").strip() or f'文本中提到了\u201c{name}\u201d这一概念。'
        out.append({"name": name, "content": fallback})
        rejected.append({"reason": "missing_written_entity_fallback", "name": name})
    return out, rejected


def dedupe_extracted_relations(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """关系去重：无向 (entity1, entity2) 字典序 + content（忽略大小写）。"""
    seen: set[Tuple[str, str, int]] = set()
    out: List[Dict[str, str]] = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        e1 = str(r.get("entity1_name") or "").strip()
        e2 = str(r.get("entity2_name") or "").strip()
        content = str(r.get("content") or "").strip()
        if not e1 or not e2 or not content:
            continue
        if e1 == e2:
            continue
        if not _is_valid_relation_content(content, e1, e2):
            continue
        n1, n2 = _normalize_pair_for_relation(e1, e2)
        key = (n1, n2, hashlib.md5(content.lower().encode()).hexdigest()[:12])
        if key in seen:
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})
    return out


def validate_extracted_relations_with_report(
    relations: Optional[List[Dict[str, Any]]],
    valid_entity_names: set[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """关系内容校验版，返回合法关系与拒绝原因。"""
    seen: set[Tuple[str, str, str]] = set()
    out: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []

    for relation in relations or []:
        if not isinstance(relation, dict):
            rejected.append({"reason": "invalid_payload"})
            continue
        entity1_name = str(relation.get("entity1_name") or "").strip()
        entity2_name = str(relation.get("entity2_name") or "").strip()
        content = str(relation.get("content") or "").strip()
        if not entity1_name or not entity2_name:
            rejected.append({
                "reason": "missing_endpoint",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        if entity1_name == entity2_name:
            rejected.append({
                "reason": "self_relation",
                "entity1_name": entity1_name,
                "entity2_name": entity2_name,
            })
            continue
        n1, n2 = _normalize_pair_for_relation(entity1_name, entity2_name)
        if n1 not in valid_entity_names or n2 not in valid_entity_names:
            rejected.append({
                "reason": "unknown_endpoint",
                "entity1_name": n1,
                "entity2_name": n2,
            })
            continue
        if not _is_valid_relation_content(content, n1, n2):
            rejected.append({
                "reason": "invalid_content",
                "entity1_name": n1,
                "entity2_name": n2,
                "content": content,
            })
            continue
        key = (n1, n2, hashlib.md5(content.lower().encode()).hexdigest()[:12])
        if key in seen:
            rejected.append({
                "reason": "duplicate_relation",
                "entity1_name": n1,
                "entity2_name": n2,
                "content": content,
            })
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})

    return out, rejected


# ---------------------------------------------------------------------------
# V1 self-consistency check
# ---------------------------------------------------------------------------

_OVER_GENERIC_NAMES: set = {
    "共同", "发展", "分离", "技术", "理论", "方法", "过程",
    "贡献", "成就", "影响", "挑战", "困难", "机会", "问题",
    "研究", "分析", "评价", "讨论", "探索", "传播", "概述",
    "总结", "背景", "特点", "意义", "价值", "地位", "方向",
    "趋势", "理念", "目标", "思想",
    "两次", "唯一", "第一位", "开创性贡献", "杰出成就",
    "国际性的奖项", "科学领域", "物理现象", "研究过程",
    "后世影响", "挑战和困难",
    "唯一", "首位",
    "女性", "丈夫", "妻子", "科学家", "物理学家", "化学家",
    "物理学家和化学家",
}

_OVER_GENERIC_PATTERNS = [
    r'^[\u4e00-\u9fff]{1,2}$',
]


def _is_self_consistent_entity(name: str, content: str, all_entities: List[Dict[str, str]],
                                source_text: str = "") -> bool:
    """检查实体的 name 和 content 是否自洽。"""
    if not name or not content:
        return False

    core_name = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
    core_name = re.sub(r'（[A-Za-z\s]+）$', '', core_name).strip()
    if name in _OVER_GENERIC_NAMES or core_name in _OVER_GENERIC_NAMES:
        return False

    if len(name) == 1 and re.match(r'^[\u4e00-\u9fff]+$', name):
        return False

    if len(name) == 2 and re.match(r'^[\u4e00-\u9fff]+$', name):
        _generic_2char = {
            "共同", "发展", "分离", "技术", "理论", "方法", "过程",
            "贡献", "成就", "影响", "挑战", "研究", "分析", "评价",
            "讨论", "探索", "传播", "概述", "总结", "背景", "特点",
            "意义", "价值", "地位", "方向", "趋势", "理念", "目标",
            "思想", "两次", "唯一", "首位", "女性", "丈夫", "妻子",
            "科学", "物理", "化学", "数学", "生物", "医学", "文学",
            "哲学", "历史", "社会", "政治", "经济", "文化", "教育",
            "技术", "工程", "艺术", "音乐", "体育", "法律", "军事",
            "商业", "工业", "农业", "环境", "能源", "材料", "信息",
            "管理", "设计", "创新", "质量", "效率", "安全", "稳定",
            "增长", "下降", "变化", "改进", "优化", "提升", "改善",
        }
        if name in _generic_2char:
            return False

    _name_in_source = False
    if source_text and len(name) >= 2:
        _name_core = re.sub(r'[(（][^)）]+[)）]', '', name).strip()
        if _name_core:
            if _name_core in source_text:
                _name_in_source = True
            elif len(_name_core) >= 3:
                _frag_size = max(2, len(_name_core) // 2)
                for i in range(len(_name_core) - _frag_size + 1):
                    _frag = _name_core[i:i+_frag_size]
                    if _frag in source_text:
                        _name_in_source = True
                        break
    if all_entities and len(content) > 20 and len(content) <= 80 and not _name_in_source:
        other_entity_names = []
        for e in all_entities:
            ename = e.get('name', '')
            if ename and ename != name:
                other_entity_names.append(ename)

        if other_entity_names:
            _mention_count = 0
            for other_name in other_entity_names:
                if len(other_name) >= 2 and other_name in content:
                    _mention_count += 1

            if _mention_count >= 2 and len(name) <= 4:
                name_mentions = content.count(name)
                if name_mentions == 0:
                    return False

    if source_text:
        _core = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
        _core = re.sub(r'（[A-Za-z\s]+）$', '', _core).strip() or name
        _core_clean = re.sub(r'[(（][^)）]+[)）]', '', _core).strip()
        if _core_clean:
            if _core_clean in source_text:
                return True
            source_lower = source_text.lower() if _core_clean.isascii() else source_text
            check_val = _core_clean.lower() if _core_clean.isascii() else _core_clean
            if len(check_val) >= 2 and check_val in source_lower:
                return True
            if not _core_clean.isascii() and len(_core_clean) >= 3:
                _frag_size = max(2, len(_core_clean) // 2)
                for i in range(len(_core_clean) - _frag_size + 1):
                    _frag = _core_clean[i:i+_frag_size]
                    if _frag in source_text:
                        return True
            if _core_clean.isascii():
                if _core_clean.lower() in source_text.lower():
                    return True
                words = _core_clean.lower().split()
                if len(words) > 1 and all(w in source_text.lower() for w in words):
                    return True
            _cn_en_aliases = {
                "Google": ["谷歌"], "IBM": ["国际商业机器"], "Microsoft": ["微软"],
                "Apple": ["苹果"], "Amazon": ["亚马逊"], "Meta": ["脸书", "元宇宙"],
                "Tesla": ["特斯拉"], "NVIDIA": ["英伟达"], "Intel": ["英特尔"],
                "OpenAI": ["开放AI"], "Quantum": ["量子"],
            }
            _aliases = _cn_en_aliases.get(_core_clean, [])
            if _aliases:
                for alias in _aliases:
                    if alias in source_text:
                        return True
            _en_in_paren = re.findall(r'[(（]([A-Za-z][A-Za-z\s]+)[)）]', name)
            for en in _en_in_paren:
                if en.strip().lower() in source_text.lower():
                    return True
            if all_entities:
                for e in all_entities:
                    if e.get('name', '') != name and _core_clean in e.get('content', ''):
                        return True
            if len(_core_clean) <= 4 and len(content) <= 40:
                return False

    return True


def _filter_self_consistent_entities(
    entities: List[Dict[str, str]],
    relations: List[Dict[str, str]],
    source_text: str = "",
) -> List[Dict[str, str]]:
    """过滤不自洽的实体。"""
    if len(entities) <= 1:
        return entities

    related_names: set = set()
    for rel in relations:
        for key in ('entity1_name', 'entity2_name'):
            n = rel.get(key, '').strip()
            if n:
                related_names.add(n)

    out: List[Dict[str, str]] = []
    _filtered = 0
    for e in entities:
        name = e.get('name', '')
        content = e.get('content', '')

        if not _is_self_consistent_entity(name, content, entities, source_text):
            if name in related_names:
                out.append(e)
            else:
                _filtered += 1
            continue

        out.append(e)

    return out
