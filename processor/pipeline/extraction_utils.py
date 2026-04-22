"""Shared extraction utilities used by both V1 and V2/V3 pipelines.

Extracted from extraction.py during Phase 2 cleanup to decouple shared
code from V1-only functions (now in _v1_legacy.py).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..models import Entity


# ---------------------------------------------------------------------------
# Entity name cleaning
# ---------------------------------------------------------------------------

def _clean_entity_name(name: str) -> str:
    """清理实体名称中的场景标注括号。

    策略：默认保留括号内容（作为消歧信息），仅在明确判定为场景标注时移除。
    场景标注的特征：包含其他实体名、地名+事件、动作片段等临时上下文信息。

    例如：
    - "曹操（汉中张鲁）" → "曹操"  （括号内容是场景/事件信息，应移除）
    - "曹操（魏王）" → "曹操（魏王）" （括号内容是身份/类型信息，应保留）
    - "杨昂（诸营）" → "杨昂" （括号内容是场景上下文，应移除）
    - "许褚（曹操亲卫）" → "许褚（曹操亲卫）" （括号内容是持久关系，应保留）
    """

    def _is_scene_annotation(content: str) -> bool:
        """判断括号内容是否是场景/事件标注（而非消歧信息）。"""
        content = content.strip()
        if not content:
            return True  # 空括号移除

        # Pure English/alphanumeric annotation → translation/alias, remove
        if re.match(r'^[A-Za-z0-9\s\-_.]+$', content):
            return True

        # 包含句号、逗号等长标点 → 事件描述，移除
        if any(c in content for c in "，。！？、；："):
            return True

        # 明确的身份/角色/关系模式 → 保留
        _identity_patterns = [
            r'[\w]+[之的里亲][\w]+',    # "XX的XX"、"XX之XX"、"XX亲XX" — 关系描述
            r'[\w]*[者师家官将相帝侯公卿]$',  # 以身份后缀结尾
            r'[\w]*[王]$',             # 以"王"结尾（但不是"张鲁"之类的名字）
            r'[\w]+[语言地点物书作品学]$',  # 以类型/领域后缀结尾（含"学"）
            r'[\d]+[-~—][\d]+',       # 时间范围 "1368-1644"
            r'第[一二三四五六七八九十百千万零\d]+[卷回章节]',  # 章节标识
            r'[\w]*[卫兵护卫从随]$',    # 以关系/从属后缀结尾（"亲卫""护卫""侍从"）
            r'[\w]*[丫头仆婢妾妃]$',   # 以家庭角色后缀结尾
        ]
        for pattern in _identity_patterns:
            if re.search(pattern, content):
                return False  # 匹配身份模式，不是场景标注

        # 短内容（≤3字）：军事/场景关键词 → 移除
        if len(content) <= 3:
            _scene_short_words = {"诸营", "中军", "前军", "后军", "左军", "右军",
                                  "大营", "前寨", "后寨", "上关", "下关"}
            if content in _scene_short_words:
                return True

        # 含军事场景关键词 → 移除
        _scene_context_keywords = [
            "营", "寨", "关", "阵", "兵", "军", "战", "守", "攻", "围",
            "破", "败", "退", "降", "逃", "追", "击", "杀", "死", "伤",
            "火起", "告免", "出马", "迎敌", "交锋",
        ]
        for kw in _scene_context_keywords:
            if kw in content:
                # 但如果同时也含关系词或身份后缀，保留
                if any(c in content for c in "的之里") or \
                   re.search(r'[者师家官将相帝侯公卿王语言地点物书作品]$', content):
                    return False
                return True

        # 以动词/动作词开头的短片段（≤6字）→ 可能是场景片段
        if len(content) <= 6:
            _verb_starts = ("众", "忽", "遂", "相", "火", "弃", "突", "失")
            if any(content.startswith(v) for v in _verb_starts):
                return True

        # 最终兜底：纯中文专有名词（2-6字），不含关系/身份标记 → 很可能是场景标注
        if re.match(r'^[\u4e00-\u9fff]{2,6}$', content):
            return True

        return False  # 默认保留

    # 处理全角括号
    def _replace_fullwidth(m):
        inner = m.group(1)
        return f"（{inner}）" if not _is_scene_annotation(inner) else ""

    cleaned = re.sub(r'（([^）]+)）', _replace_fullwidth, name)

    # 处理半角括号
    def _replace_halfwidth(m):
        inner = m.group(1)
        return f"({inner})" if not _is_scene_annotation(inner) else ""

    cleaned = re.sub(r'\(([^)]+)\)', _replace_halfwidth, cleaned)

    # Handle "/" compound names: "变更/Mutation" → "变更" (strip English alias)
    if "/" in cleaned:
        parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(parts) == 2:
            _english_re = re.compile(r'^[A-Za-z0-9\s\-_.]+$')
            if _english_re.match(parts[1]) and not _english_re.match(parts[0]):
                cleaned = parts[0]
            elif _english_re.match(parts[0]) and not _english_re.match(parts[1]):
                cleaned = parts[1]

    return cleaned.strip() or name.strip()


# ---------------------------------------------------------------------------
# Relation content minimum length
# ---------------------------------------------------------------------------

# 关系内容最小字符数 — extraction 和 relation 模块统一使用
MIN_RELATION_CONTENT_LENGTH = 8


# ---------------------------------------------------------------------------
# Content-similarity dedup
# ---------------------------------------------------------------------------

def _dedupe_by_content_similarity(
    entities: List[Dict[str, str]],
    relations: List[Dict[str, str]],
    jaccard_threshold: float = 0.65,
) -> List[Dict[str, str]]:
    """按内容相似度去重：如果多个实体的content高度相似，说明它们都在描述同一主体而非自身。"""
    if len(entities) <= 1:
        return entities

    # 收集关系端点，用于判断实体是否"有连接"
    related_names: set = set()
    for rel in relations:
        for key in ('entity1_name', 'entity2_name'):
            name = rel.get(key, '').strip()
            if name:
                related_names.add(name)

    def _word_set(text: str) -> set:
        words = set()
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                words.add(char)
            elif char.isalnum():
                words.add(char.lower())
        return words

    content_words = [_word_set(e.get('content', '')) for e in entities]

    # Union-Find 分组
    parent = list(range(len(entities)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(entities)):
        if not content_words[i]:
            continue
        for j in range(i + 1, len(entities)):
            if not content_words[j]:
                continue
            intersection = content_words[i] & content_words[j]
            union_set = content_words[i] | content_words[j]
            if union_set:
                jaccard = len(intersection) / len(union_set)
                if jaccard > jaccard_threshold:
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(entities)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    out: List[Dict[str, str]] = []
    for indices in groups.values():
        if len(indices) == 1:
            out.append(entities[indices[0]])
            continue
        # 多个实体内容高度相似：保留最有价值的一个
        def _entity_score(idx: int) -> tuple:
            name = entities[idx].get('name', '')
            is_related = name in related_names
            content_len = len(entities[idx].get('content', ''))
            return (is_related, content_len)

        best_idx = max(indices, key=_entity_score)
        out.append(entities[best_idx])

    return out


# ---------------------------------------------------------------------------
# Combined entity + relation dedup (used by orchestrator cache loader)
# ---------------------------------------------------------------------------

def dedupe_extraction_lists(
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """供缓存加载等场景：实体、关系各做一次列表级去重。

    委托给 _v1_legacy 中的去重函数，以避免循环依赖。
    """
    from ._v1_legacy import dedupe_extracted_entities, dedupe_extracted_relations
    return dedupe_extracted_entities(entities), dedupe_extracted_relations(relations)


# ---------------------------------------------------------------------------
# Alignment result dataclass
# ---------------------------------------------------------------------------

@dataclass
class _AlignResult:
    """步骤6（实体对齐）的输出，供步骤7使用。"""
    entity_name_to_id: Dict[str, str] = field(default_factory=dict)
    pending_relations: List[Dict] = field(default_factory=list)
    unique_entities: List[Entity] = field(default_factory=list)
    unique_pending_relations: List[Dict] = field(default_factory=list)
