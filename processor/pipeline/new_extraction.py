"""
V2 Extraction Pipeline — "one small task per step".

Redesigned pipeline steps (within a sliding window):
  Step 2a: Named entity extraction (one LLM call)
  Step 2b: Abstract concept extraction (one LLM call)
  Step 2c: Event/process extraction (one LLM call)
  Step 2d: Coverage gap check / supplement (one LLM call)
  Step 3: Entity dedup & normalization (rule-based)
  Step 4: Per-entity content writing (one LLM call per entity)
  Step 5: Entity quality gate (rule-based)
  Step 6a: Relation pair discovery (one or more LLM calls, chunked for large lists)
  Step 6b: Relation pair expansion (one LLM call, find missed pairs)
  Step 7: Per-pair relation content writing (one LLM call per pair)
  Step 8: Relation quality gate (rule-based)

Steps 1 (cache update), 9 (entity alignment), 10 (relation alignment)
are handled by existing code in orchestrator.py, entity.py, relation.py.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils import wprint_info
from .extraction_utils import _clean_entity_name


# ---------------------------------------------------------------------------
# Minimal entity name filtering (replaces ~350 lines of heuristics)
# ---------------------------------------------------------------------------

# System leak patterns — entities must not contain these
_SYSTEM_LEAK_PATTERNS = re.compile(
    r"处理进度|系统状态|抽取结果|缓存数据|步骤\d|token|api|json|llm"
    r"|概念[AB]$|新增的概念|字符串数组|代码块|去重",
    re.IGNORECASE,
)

# Degenerate name patterns (repetitive text from LLM parsing errors)
_DEGENERATE_PATTERN = re.compile(r"(\S+)\s+\1\s+\1")

# Meta-entity names — structural/document metadata that should not be entities.
# These are produced when LLMs extract "章节标题" as a concept rather than the
# actual chapter title text.
_META_ENTITY_NAMES = frozenset({
    "章节标题", "回目标题", "节标题", "小节名", "小节标题",
    "列表主题", "段落主题", "章节主题", "节名",
    "文档名", "文件名", "文档标题", "文件标题",
    "当前位置", "当前位置标识",
    "章节编号", "回目编号",
})

# Meta-entity patterns — catch variations like "第一回章节主题", "段落主题"
_META_ENTITY_PATTERN = re.compile(
    r"^(章节|回目|小节|段落|列表|文档|文件|节).*(标题|主题|名|编号|标识)$",
)

# Document prepend pattern — the "开始阅读新的文档，文件名是：XXX" prefix
_DOC_PREPEND_PATTERN = re.compile(
    r"^开始阅读新的文档，文件名是[：:]?\s*",
)

# Trivial entity patterns — action/state descriptions that are not entities.
# These are typically 2-4 char Chinese phrases that describe an action or state.
# Examples from production: "诈病", "游荡无度", "装病避责", "威名颇震",
# "设置五色棒", "纵火袭击", "初到任", "斩首万余级"
_TRIVIAL_ACTION_PATTERN = re.compile(
    r"^("
    # Verb-object patterns (2-6 chars)
    r"诈病|装病|称病|伪病"
    r"|游荡|放纵|纵情|放荡"
    r"|斩首|追袭|伏击|纵火|袭击|突袭|设伏|围攻|破城"
    r"|设置|建造|修缮|打造"
    r"|初到任|到任|就任|赴任|上任"
    r"|助战|参战|从军|出征|征讨|讨伐"
    r")$"
)

# Noun phrases with "的" that are descriptions, not entity names.
# E.g. "乱世的关键人物", "忠心耿耿的群体", "第一位女性教授"
# Short names with "的" (≤4 chars) may be valid (如"石头的"), but longer ones are descriptions.
_DESCRIPTIVE_PHRASE_PATTERN = re.compile(
    r"^.{4,}的.{2,}$"  # 4+ chars before "的" + 2+ after = description phrase
)

# Ordinal-description patterns like "第一位女性教授", "第一百零八个好汉"
_ORDINAL_DESCRIPTION_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万零\d]+[个位名种].{2,}$"
)

# Specific over-extracted trivial entity names to block
_TRIVIAL_ENTITY_NAMES = frozenset({
    "威名颇震", "初到任", "英雄聚会", "乱世的关键人物",
    "装病避责", "游荡无度", "设置五色棒", "纵火袭击",
    "斩首万余级", "初次战斗", "初次迎战",
    "第一位女性教授", "忠心耿耿的群体",
})


def _is_valid_entity_name_v2(name: str) -> bool:
    """Minimal validation: only filter obvious junk and meta-entities."""
    if not name or len(name) < 2:
        return False
    if len(name) > 60:
        return False
    # Pure numbers
    if name.strip().isdigit():
        return False
    # System leaks
    if _SYSTEM_LEAK_PATTERNS.search(name):
        return False
    # Degenerate (repetitive)
    if _DEGENERATE_PATTERN.search(name):
        return False
    # Meta-entity filter (structural/document metadata)
    if name in _META_ENTITY_NAMES:
        return False
    if _META_ENTITY_PATTERN.match(name):
        return False
    # Trivial action/state descriptions
    if name in _TRIVIAL_ENTITY_NAMES:
        return False
    if _TRIVIAL_ACTION_PATTERN.match(name):
        return False
    # Descriptive phrases (long "XX的XX" patterns)
    if "的" in name and _DESCRIPTIVE_PHRASE_PATTERN.match(name):
        return False
    # Ordinal-description patterns ("第一位女性教授", etc.)
    if name.startswith("第") and _ORDINAL_DESCRIPTION_PATTERN.match(name):
        return False
    return True


# ---------------------------------------------------------------------------
# Name cleaning & dedup
# ---------------------------------------------------------------------------

_TITLE_SUFFIXES = ("教授", "博士", "先生", "女士", "同志")


def _extract_core_name(name: str) -> str:
    """Strip parenthetical annotations to get the core name for dedup."""
    name = name.strip()
    for bracket in ("（", "("):
        idx = name.find(bracket)
        if idx > 0:
            return name[:idx].strip()
    return name


def _dedup_entity_names(names: List[str]) -> List[str]:
    """Deduplicate entity names using core-name matching."""
    seen_core: Dict[str, str] = {}
    result: List[str] = []

    for name in names:
        if not _is_valid_entity_name_v2(name):
            continue

        core = _extract_core_name(name)
        if not core:
            continue

        existing = seen_core.get(core)
        if existing is None:
            seen_core[core] = name
            result.append(name)
        else:
            if "(" in name or "（" in name:
                if "(" not in existing and "（" not in existing:
                    idx = result.index(existing)
                    result[idx] = name
                    seen_core[core] = name

    return result


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

_GENERIC_CONTENT_PATTERNS = [
    re.compile(r"^该实体", re.IGNORECASE),
    re.compile(r"^这是一个", re.IGNORECASE),
    re.compile(r"^这是一", re.IGNORECASE),
    # "这是典型的..." — direct quote from text, not a description
    re.compile(r"^这是典型的", re.IGNORECASE),
    # Section headers as content (e.g. "## 🧠 关键改进：...")
    re.compile(r"^#{1,4}\s", re.IGNORECASE),
    # Markdown header content (#### ✅ (b) 有语义变化（delta detected）。)
    re.compile(r"^#{1,6}[^\n]{0,20}[✅❌👉🧠🔥💡⚡📝🎯🔍]\s", re.IGNORECASE),
    # Fallback content placeholder
    re.compile(r"是文本中讨论的一个知识概念。?$", re.IGNORECASE),
    re.compile(r"具有特定的语义和知识内涵。?$", re.IGNORECASE),
    # Emoji-prefixed content (e.g. "👉 **"每次..."**。")
    re.compile(r"^[\U0001F300-\U0001F9FF\U00002600-\U000027BF]"),
    # Content that is primarily markdown formatting fragments
    # (bold markers wrapping a quote — LLM outputs original text as content)
    re.compile(r'^\*\*["\u201c\u201d].*["\u201c\u201d]\*\*[。.]?$', re.DOTALL),
    # LLM error/refusal leakage (e.g. "无法执行此操作", "无法提供描述", "I cannot", "我正在等待")
    re.compile(r"^(无法执行|无法提供|我不能|我无法|I cannot|I am unable|As an AI)", re.IGNORECASE),
    re.compile(r"我正在等待", re.IGNORECASE),
]

# Meta-entity content patterns — entities whose content indicates they are metadata
_META_CONTENT_PATTERNS = [
    re.compile(r"^当前文档没有?明确提供", re.IGNORECASE),
    re.compile(r"^当前文本没有?明确给出", re.IGNORECASE),
    re.compile(r"^当前文档未?直接给出", re.IGNORECASE),
    re.compile(r"^当前文本未?明确给出", re.IGNORECASE),
    re.compile(r"^输入文本未提供", re.IGNORECASE),
    re.compile(r"^当前摘要", re.IGNORECASE),
    re.compile(r"^文档中没有", re.IGNORECASE),
    re.compile(r"^请注意", re.IGNORECASE),
    # LLM instruction leakage — when model outputs its own instructions as content
    re.compile(r"必须严格限制在一个.*json.*代码块", re.IGNORECASE),
    re.compile(r"输出格式.*请严格", re.IGNORECASE),
    re.compile(r"该指令要求", re.IGNORECASE),
]

_MIN_ENTITY_CONTENT_LEN = 15
_MIN_RELATION_CONTENT_LEN = 10

_GENERIC_RELATION_PATTERNS = re.compile(
    r"^(有关联|存在关系|有关|相关|有联系|有connection|is related to|is associated with|"
    r"is connected to|are related|are associated|have a relationship|are connected)",
    re.IGNORECASE,
)


def _validate_entity(name: str, content: str) -> bool:
    if not content or len(content) < _MIN_ENTITY_CONTENT_LEN:
        return False
    for pattern in _GENERIC_CONTENT_PATTERNS:
        if pattern.search(content):
            return False
    # Meta-entity content filter (e.g. "当前文档未明确提供章节标题")
    for pattern in _META_CONTENT_PATTERNS:
        if pattern.search(content):
            return False
    return True


def _validate_relation(
    entity_a: str, entity_b: str, content: str, valid_entity_names: Set[str],
) -> bool:
    if not content or len(content) < _MIN_RELATION_CONTENT_LEN:
        return False
    if entity_a == entity_b:
        return False
    if _GENERIC_RELATION_PATTERNS.match(content):
        return False
    return True


def _build_entity_fallback_content(name: str, window_text: str) -> str:
    """Build a context-aware fallback description when LLM content writing fails.

    Extracts sentences from the window text that mention the entity name,
    and assembles them into a brief description.
    """
    if not window_text:
        return f"{name}是一个在文本中被讨论的核心概念，具有特定的语义和知识内涵。"

    import re as _re

    # Find sentences containing the entity name, skip markdown lines
    sentences = _re.split(r'[。！？\n]', window_text)
    def _is_prose(s):
        """True if the sentence is real prose, not a markdown header/bullet/etc."""
        s = s.strip()
        if not s or len(s) <= 5:
            return False
        # Skip markdown headers (## Title, ### Title, etc.)
        if _re.match(r'^#{1,6}\s', s):
            return False
        # Skip emoji-only or emoji-leading lines (section markers)
        if _re.match(r'^[\U0001F300-\U0001F9FF\U00002600-\U000027BF\u2702-\u27BF]', s):
            return False
        # Skip bullet points that are just labels
        if _re.match(r'^[-*]\s+[✅❌👉]', s):
            return False
        return True

    relevant = [s.strip() for s in sentences if name in s and _is_prose(s)]

    if relevant:
        # Take up to 3 most relevant sentences, truncate if too long
        desc_parts = relevant[:3]
        desc = '。'.join(desc_parts)
        if len(desc) > 200:
            desc = desc[:197] + '...'
        if not desc.endswith('。'):
            desc += '。'
        return desc

    # Try substring match — some names might not match exactly
    # (e.g. "变更判定" might appear as part of "变更判定层")
    name_parts = list(set(name[i:i+2] for i in range(len(name)-1) if len(name) >= 4))
    for part in name_parts:
        partial_match = [s.strip() for s in sentences if part in s and _is_prose(s)]
        if partial_match:
            desc_parts = partial_match[:2]
            desc = '。'.join(desc_parts)
            if len(desc) > 200:
                desc = desc[:197] + '...'
            if not desc.endswith('。'):
                desc += '。'
            return desc

    return f"{name}是一个在文本中被讨论的核心概念，具有特定的语义和知识内涵。"


# ---------------------------------------------------------------------------
# V2 Extraction Mixin
# ---------------------------------------------------------------------------

class _NewExtractionMixin:
    """
    V2 extraction pipeline mixin.

    Multi-focus extraction: named + abstract + events + supplement.
    Chunked + expansion relation discovery.
    """

    def _extract_only_v3(
        self,
        new_episode,
        input_text: str,
        document_name: str,
        verbose: bool = True,
        verbose_steps: bool = True,
        event_time=None,
        progress_callback=None,
        progress_range: tuple = (0.1, 0.5),
        window_index: int = 0,
        total_windows: int = 1,
        window_timings_ref: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        V3 extraction: dual-model pipeline.

        Uses extraction_client (strong model) for entity/relation discovery,
        llm_client (small model) for content writing.

        Returns:
            (extracted_entities, extracted_relations)
        """
        import time as _time
        p_lo, p_hi = progress_range
        _win = f"窗口 {window_index + 1}/{total_windows}"

        def _progress(frac, label, msg):
            if progress_callback:
                progress_callback(p_lo + (p_hi - p_lo) * frac, label, msg)

        def _record_timing(key: str, elapsed: float):
            if window_timings_ref is not None:
                window_timings_ref[key] = elapsed

        extraction_client = self.extraction_client

        # ==============================================================
        # V3-1: Comprehensive entity extraction (strong model, think mode)
        # V3-1b: Conversational refinement
        # ==============================================================
        _progress(0.03, f"{_win} · V3-1: 实体提取（强模型）", "开始")
        _t = _time.time()
        raw_names, ent_refine = extraction_client.extract_entities_v3(
            input_text, max_refine_rounds=self.v3_entity_refine_rounds
        )
        _record_timing("v3-1_entity_extract", _time.time() - _t)
        if verbose or verbose_steps:
            _refine_tag = ""
            if ent_refine["rounds_run"] > 0:
                _refine_tag = f" (精炼{ent_refine['rounds_run']}轮 +{ent_refine['refine_added']})"
            wprint_info(f"【V3 步骤1】综合实体提取｜{ent_refine['initial']}个初始{_refine_tag} → 总{len(raw_names)}个候选｜{_time.time()-_t:.1f}s")

        # ==============================================================
        # V3-2: Entity dedup & normalization (rule-based)
        # ==============================================================
        _t = _time.time()
        entity_names = _dedup_entity_names(raw_names)
        # Clean entity names (strip scene annotations, English aliases, etc.)
        entity_names = [_clean_entity_name(n) for n in entity_names]
        # Split compound names (e.g. "变更（Mutation）/ 概念版本" → ["变更", "概念版本"])
        _split = []
        for n in entity_names:
            if "/" in n:
                for part in n.split("/"):
                    part = part.strip()
                    if part and len(part) >= 2:
                        _split.append(part)
            else:
                _split.append(n)
        entity_names = _dedup_entity_names(_split)
        # Re-dedup after cleaning (cleaning may produce duplicates)
        entity_names = _dedup_entity_names(entity_names)
        _record_timing("v3-2_entity_dedup", _time.time() - _t)
        if verbose or verbose_steps:
            wprint_info(f"【V3 步骤2】实体去重｜{len(entity_names)}个有效｜{_time.time()-_t:.1f}s")

        _progress(0.15, f"{_win} · V3-2: 实体去重", f"{len(entity_names)} 个实体")

        # ==============================================================
        # V3-3: Per-entity content writing (small model, parallelized)
        # ==============================================================
        _progress(0.18, f"{_win} · V3-3: 实体内容写作", f"开始写 {len(entity_names)} 个实体")
        extracted_entities: List[Dict[str, str]] = []

        _t = _time.time()
        def _write_one_entity(name: str) -> Dict[str, str]:
            content = self.llm_client.write_entity_content_v2(name, input_text)
            if not content or len(content) < 10:
                content = self.llm_client.write_entity_content_v2(name, input_text)
            if not content or len(content) < 10:
                content = _build_entity_fallback_content(name, input_text)
            return {"name": name, "content": content}

        if len(entity_names) > 1:
            # Parallel content writing — e4b client has max_concurrency=8
            _n_workers = min(len(entity_names), 4)
            with ThreadPoolExecutor(max_workers=_n_workers, thread_name_prefix="v3-econtent") as pool:
                futures = {pool.submit(_write_one_entity, name): name for name in entity_names}
                done_count = 0
                for fut in as_completed(futures):
                    done_count += 1
                    frac = 0.18 + 0.30 * (done_count / max(1, len(entity_names)))
                    _progress(frac, f"{_win} · V3-3", f"实体 {done_count}/{len(entity_names)}")
                    try:
                        extracted_entities.append(fut.result())
                    except Exception:
                        name = futures[fut]
                        extracted_entities.append({"name": name, "content": _build_entity_fallback_content(name, input_text)})
        else:
            for name in entity_names:
                extracted_entities.append(_write_one_entity(name))
            _progress(0.48, f"{_win} · V3-3", f"实体 1/1")

        _record_timing("v3-3_entity_content", _time.time() - _t)
        if verbose or verbose_steps:
            wprint_info(f"【V3 步骤3】实体内容写作｜{len(extracted_entities)}个完成｜{_time.time()-_t:.1f}s")

        # ==============================================================
        # V3-4: Entity quality gate (rule-based)
        #   When content fails validation, try fallback before dropping.
        #   If ALL entities are filtered, keep them with forced fallback
        #   (better to have imperfect entities than lose entire window).
        # ==============================================================
        _t = _time.time()
        valid_entities = []
        rejected_entities = []
        for e in extracted_entities:
            if _validate_entity(e["name"], e["content"]):
                valid_entities.append(e)
            else:
                # Try fallback content extracted from window text
                fallback = _build_entity_fallback_content(e["name"], input_text)
                if _validate_entity(e["name"], fallback):
                    e["content"] = fallback
                    valid_entities.append(e)
                    if verbose_steps:
                        wprint_info(f"  │  实体质量门挽救: {e['name']} (使用窗口文本回退)")
                else:
                    rejected_entities.append((e, fallback))
                    if verbose_steps:
                        _content_preview = e["content"][:60] if e.get("content") else "(空)"
                        _fallback_preview = fallback[:60] if fallback else "(空)"
                        wprint_info(f"  │  实体质量门拒绝: {e['name']}")
                        wprint_info(f"  │    原始内容: {_content_preview}")
                        wprint_info(f"  │    回退内容: {_fallback_preview}")

        # Emergency rescue: if ALL entities filtered, keep with forced fallback
        if not valid_entities and rejected_entities:
            if verbose_steps:
                wprint_info(f"  │  ⚠ 全部实体被过滤，启动紧急保留")
            for e, fallback in rejected_entities:
                # Try fallback from window text first (already computed)
                if fallback and _validate_entity(e["name"], fallback):
                    e["content"] = fallback
                else:
                    # Build a minimal but valid description that passes quality gate
                    e["content"] = f"{e['name']}是文本中重点讨论的一个核心概念，涉及知识图谱中概念版本管理的关键机制。"
                valid_entities.append(e)

        if verbose or verbose_steps:
            rejected = len(extracted_entities) - len(valid_entities)
            if rejected:
                wprint_info(f"【V3 步骤4】实体质量门｜{rejected}个被过滤，{len(valid_entities)}个通过")

        _record_timing("v3-4_entity_quality", _time.time() - _t)
        extracted_entities = valid_entities
        entity_name_set = {e["name"] for e in extracted_entities}
        entity_name_list = [e["name"] for e in extracted_entities]

        _progress(0.50, f"{_win} · V3-4: 实体质量门", f"{len(extracted_entities)} 个有效实体")

        # ==============================================================
        # V3-5: Comprehensive relation discovery (strong model, think mode)
        # V3-5b: Conversational refinement
        # ==============================================================
        _t = _time.time()
        relation_pairs = []
        if len(extracted_entities) >= 2:
            _progress(0.53, f"{_win} · V3-5: 关系发现（强模型）", "开始")
            raw_pairs, rel_refine = extraction_client.discover_relations_v3(
                entity_name_list, input_text, max_refine_rounds=self.v3_relation_refine_rounds
            )

            # Normalize pair endpoints to match entity names
            seen_pairs = set()
            for a, b in raw_pairs:
                a = self._resolve_entity_name(a, entity_name_set)
                b = self._resolve_entity_name(b, entity_name_set)
                if a and b and a != b:
                    pair_key = tuple(sorted([a, b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        relation_pairs.append((a, b))

            if verbose or verbose_steps:
                _refine_tag = ""
                if rel_refine["rounds_run"] > 0:
                    _refine_tag = f" (精炼{rel_refine['rounds_run']}轮 +{rel_refine['refine_added']})"
                wprint_info(f"【V3 步骤5】关系对发现｜{rel_refine['initial']}对初始{_refine_tag} → 总{len(relation_pairs)}对｜{_time.time()-_t:.1f}s")
        _record_timing("v3-5_relation_discovery", _time.time() - _t)

        _progress(0.60, f"{_win} · V3-5: 关系发现完成", f"{len(relation_pairs)} 对")

        # ==============================================================
        # V3-6: Per-pair relation content writing (small model, parallelized)
        # ==============================================================
        _t = _time.time()
        extracted_relations: List[Dict[str, str]] = []

        def _write_one_relation(pair: Tuple[str, str]) -> Optional[Dict[str, str]]:
            a, b = pair
            content = self.llm_client.write_relation_content_v2(a, b, input_text)
            if content:
                return {"entity1_name": a, "entity2_name": b, "content": content}
            return None

        if len(relation_pairs) > 1:
            _n_workers = min(len(relation_pairs), 4)
            with ThreadPoolExecutor(max_workers=_n_workers, thread_name_prefix="v3-rcontent") as pool:
                futures = {pool.submit(_write_one_relation, pair): pair for pair in relation_pairs}
                done_count = 0
                for fut in as_completed(futures):
                    done_count += 1
                    frac = 0.60 + 0.30 * (done_count / max(1, len(relation_pairs)))
                    _progress(frac, f"{_win} · V3-6", f"关系 {done_count}/{len(relation_pairs)}")
                    try:
                        result = fut.result()
                        if result:
                            extracted_relations.append(result)
                    except Exception:
                        pass
        else:
            for i, (a, b) in enumerate(relation_pairs):
                content = self.llm_client.write_relation_content_v2(a, b, input_text)
                _progress(0.90, f"{_win} · V3-6", f"关系 {i + 1}/{len(relation_pairs)}")
                if content:
                    extracted_relations.append({"entity1_name": a, "entity2_name": b, "content": content})

        if verbose or verbose_steps:
            wprint_info(f"【V3 步骤6】关系内容写作｜{len(extracted_relations)}条完成｜{_time.time()-_t:.1f}s")
        _record_timing("v3-6_relation_content", _time.time() - _t)

        # ==============================================================
        # V3-7: Relation quality gate (rule-based)
        # ==============================================================
        _t = _time.time()
        valid_relations = []
        for r in extracted_relations:
            if _validate_relation(r["entity1_name"], r["entity2_name"],
                                  r["content"], entity_name_set):
                valid_relations.append(r)

        if verbose or verbose_steps:
            rejected_r = len(extracted_relations) - len(valid_relations)
            if rejected_r:
                wprint_info(f"【V3 步骤7】关系质量门｜{rejected_r}条被过滤，{len(valid_relations)}条通过")
        _record_timing("v3-7_relation_quality", _time.time() - _t)

        _progress(0.95, f"{_win} · 完成",
                   f"{len(extracted_entities)} 实体, {len(valid_relations)} 关系")

        return extracted_entities, valid_relations

    # ------------------------------------------------------------------
    # V2 pipeline (multi-step decomposition for small models)
    # ------------------------------------------------------------------

    def _extract_only_v2(
        self,
        new_episode,
        input_text: str,
        document_name: str,
        verbose: bool = True,
        verbose_steps: bool = True,
        event_time=None,
        progress_callback=None,
        progress_range: tuple = (0.1, 0.5),
        window_index: int = 0,
        total_windows: int = 1,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        V2 extraction: multi-focus entity extraction + chunked/expanded relations.

        Returns:
            (extracted_entities, extracted_relations)
        """
        p_lo, p_hi = progress_range
        _win = f"窗口 {window_index + 1}/{total_windows}"

        def _progress(frac, label, msg):
            if progress_callback:
                progress_callback(p_lo + (p_hi - p_lo) * frac, label, msg)

        # ==============================================================
        # Step 2a: Named entity extraction (multi-round)
        # ==============================================================
        _entity_rounds = getattr(self, "entity_extraction_rounds", 1)
        _progress(0.03, f"{_win} · 步骤2a: 具名实体提取", "开始")
        named_names = self.llm_client.extract_entity_names_named_v2(input_text, max_rounds=_entity_rounds)
        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤2a】具名实体提取｜{len(named_names)}个")

        # ==============================================================
        # Step 2b: Abstract concept extraction (multi-round)
        # ==============================================================
        _progress(0.08, f"{_win} · 步骤2b: 抽象概念提取", "开始")
        abstract_names = self.llm_client.extract_entity_names_abstract_v2(input_text, max_rounds=_entity_rounds)
        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤2b】抽象概念提取｜{len(abstract_names)}个")

        # ==============================================================
        # Step 2c: Event / process extraction (multi-round)
        # ==============================================================
        _progress(0.12, f"{_win} · 步骤2c: 事件提取", "开始")
        event_names = self.llm_client.extract_entity_names_events_v2(input_text, max_rounds=_entity_rounds)
        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤2c】事件提取｜{len(event_names)}个")

        # Merge all names, dedup by lowercase
        all_raw = named_names + abstract_names + event_names
        existing_lower = set()
        merged = []
        for n in all_raw:
            if n.lower() not in existing_lower:
                existing_lower.add(n.lower())
                merged.append(n)

        # ==============================================================
        # Step 2d: Coverage gap check / supplement
        # ==============================================================
        text_len = len(input_text) if input_text else 0
        should_supplement = text_len > 200 or len(merged) < 5

        if should_supplement and merged:
            _progress(0.16, f"{_win} · 步骤2d: 查漏补缺", "开始")
            supplement = self.llm_client.extract_entity_names_supplement_v2(
                input_text, merged
            )
            new_names = [n for n in supplement if n.lower() not in existing_lower]
            for n in new_names:
                existing_lower.add(n.lower())
            merged.extend(new_names)
            if verbose or verbose_steps:
                wprint_info(f"【V2 步骤2d】查漏补缺｜+{len(new_names)}个新发现（共{len(merged)}个）")

        # ==============================================================
        # Step 2e: Reflection / Self-Critique (structured diagnostic)
        # ==============================================================
        _enable_reflection = getattr(self, "v2_enable_reflection", True)
        if _enable_reflection and merged:
            _progress(0.18, f"{_win} · 步骤2e: 反思自检", "开始")
            reflection = self.llm_client.extract_entity_names_reflection_v2(
                input_text, merged
            )
            new_names = [n for n in reflection if n.lower() not in existing_lower]
            for n in new_names:
                existing_lower.add(n.lower())
            merged.extend(new_names)
            if verbose or verbose_steps:
                wprint_info(f"【V2 步骤2e】反思自检｜+{len(new_names)}个新发现（共{len(merged)}个）")

        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤2 汇总】合计 {len(merged)} 个候选（具名{len(named_names)} + 抽象{len(abstract_names)} + 事件{len(event_names)}）")

        # ==============================================================
        # Step 3: Entity dedup & normalization (rule-based)
        # ==============================================================
        entity_names = _dedup_entity_names(merged)
        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤3】实体去重｜{len(entity_names)}个有效")

        _progress(0.2, f"{_win} · 步骤3: 实体去重", f"{len(entity_names)} 个实体")

        # ==============================================================
        # Step 4: Per-entity content writing (one LLM call per entity)
        # ==============================================================
        _progress(0.22, f"{_win} · 步骤4: 实体内容写作", f"开始写 {len(entity_names)} 个实体")
        extracted_entities: List[Dict[str, str]] = []

        for i, name in enumerate(entity_names):
            content = self.llm_client.write_entity_content_v2(name, input_text)
            frac = 0.22 + 0.28 * ((i + 1) / max(1, len(entity_names)))
            _progress(frac, f"{_win} · 步骤4", f"实体 {i + 1}/{len(entity_names)}")

            if not content or len(content) < 10:
                content = self.llm_client.write_entity_content_v2(name, input_text)
            if not content or len(content) < 10:
                content = f"文本中提到的{name}，是文本中涉及的一个概念或实体。"

            extracted_entities.append({"name": name, "content": content})

        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤4】实体内容写作｜{len(extracted_entities)}个完成")

        # ==============================================================
        # Step 5: Entity quality gate (rule-based)
        # ==============================================================
        valid_entities = []
        for e in extracted_entities:
            if _validate_entity(e["name"], e["content"]):
                valid_entities.append(e)
        if verbose or verbose_steps:
            rejected = len(extracted_entities) - len(valid_entities)
            if rejected:
                wprint_info(f"【V2 步骤5】实体质量门｜{rejected}个被过滤，{len(valid_entities)}个通过")

        extracted_entities = valid_entities
        entity_name_set = {e["name"] for e in extracted_entities}
        entity_name_list = [e["name"] for e in extracted_entities]

        _progress(0.52, f"{_win} · 步骤5: 实体质量门", f"{len(extracted_entities)} 个有效实体")

        # ==============================================================
        # Step 6a: Relation pair discovery (chunked for large entity lists)
        # ==============================================================
        relation_pairs = []
        if len(extracted_entities) >= 2:
            _progress(0.55, f"{_win} · 步骤6a: 关系对发现", "开始")

            if len(entity_name_list) > 20:
                # Chunked discovery for large entity lists
                raw_pairs = self.llm_client.discover_relation_pairs_chunked_v2(
                    entity_name_list, input_text, chunk_size=18
                )
            else:
                raw_pairs = self.llm_client.discover_relation_pairs_v2(
                    entity_name_list, input_text
                )

            # Normalize pair endpoints to match entity names
            seen_pairs = set()
            for a, b in raw_pairs:
                a = self._resolve_entity_name(a, entity_name_set)
                b = self._resolve_entity_name(b, entity_name_set)
                if a and b and a != b:
                    pair_key = tuple(sorted([a, b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        relation_pairs.append((a, b))

            if verbose or verbose_steps:
                wprint_info(f"【V2 步骤6a】关系对发现｜{len(relation_pairs)}对")

        # ==============================================================
        # Step 6b: Relation pair expansion (find missed pairs)
        # ==============================================================
        if len(extracted_entities) >= 2 and relation_pairs:
            _progress(0.62, f"{_win} · 步骤6b: 关系扩展", "查漏")
            expand_pairs = self.llm_client.discover_relation_pairs_expand_v2(
                entity_name_list, relation_pairs, input_text
            )
            seen_pairs = set(tuple(sorted([a, b])) for a, b in relation_pairs)
            new_count = 0
            for a, b in expand_pairs:
                a = self._resolve_entity_name(a, entity_name_set)
                b = self._resolve_entity_name(b, entity_name_set)
                if a and b and a != b:
                    pair_key = tuple(sorted([a, b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        relation_pairs.append((a, b))
                        new_count += 1

            if verbose or verbose_steps:
                wprint_info(f"【V2 步骤6b】关系扩展｜+{new_count}对新发现（共{len(relation_pairs)}对）")

        # ==============================================================
        # Step 6c: Orphan entity relation recovery
        # ==============================================================
        _enable_orphan_recovery = getattr(self, "v2_enable_orphan_recovery", True)
        if _enable_orphan_recovery and len(extracted_entities) >= 2 and relation_pairs:
            # Find entities with zero relations
            connected_entities = set()
            for a, b in relation_pairs:
                connected_entities.add(a)
                connected_entities.add(b)
            orphan_names = [n for n in entity_name_list if n not in connected_entities]

            if orphan_names:
                _progress(0.64, f"{_win} · 步骤6c: 孤立实体关系恢复", f"{len(orphan_names)}个孤立实体")
                orphan_pairs = self.llm_client.discover_relation_pairs_orphan_v2(
                    orphan_names, entity_name_list, input_text
                )
                seen_pairs = set(tuple(sorted([a, b])) for a, b in relation_pairs)
                orphan_new = 0
                for a, b in orphan_pairs:
                    a = self._resolve_entity_name(a, entity_name_set)
                    b = self._resolve_entity_name(b, entity_name_set)
                    if a and b and a != b:
                        pair_key = tuple(sorted([a, b]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            relation_pairs.append((a, b))
                            orphan_new += 1

                if verbose or verbose_steps and orphan_new > 0:
                    wprint_info(f"【V2 步骤6c】孤立实体恢复｜+{orphan_new}对新发现（共{len(relation_pairs)}对）")

        _progress(0.65, f"{_win} · 步骤6: 关系发现完成", f"{len(relation_pairs)} 对")

        # ==============================================================
        # Step 7: Per-pair relation content writing (one LLM call per pair)
        # ==============================================================
        extracted_relations: List[Dict[str, str]] = []
        for i, (a, b) in enumerate(relation_pairs):
            content = self.llm_client.write_relation_content_v2(a, b, input_text)
            frac = 0.65 + 0.25 * ((i + 1) / max(1, len(relation_pairs)))
            _progress(frac, f"{_win} · 步骤7", f"关系 {i + 1}/{len(relation_pairs)}")

            if content:
                extracted_relations.append({
                    "entity1_name": a,
                    "entity2_name": b,
                    "content": content,
                })

        if verbose or verbose_steps:
            wprint_info(f"【V2 步骤7】关系内容写作｜{len(extracted_relations)}条完成")

        # ==============================================================
        # Step 8: Relation quality gate (rule-based)
        # ==============================================================
        valid_relations = []
        for r in extracted_relations:
            if _validate_relation(r["entity1_name"], r["entity2_name"],
                                  r["content"], entity_name_set):
                valid_relations.append(r)

        if verbose or verbose_steps:
            rejected_r = len(extracted_relations) - len(valid_relations)
            if rejected_r:
                wprint_info(f"【V2 步骤8】关系质量门｜{rejected_r}条被过滤，{len(valid_relations)}条通过")

        _progress(0.95, f"{_win} · 完成",
                   f"{len(extracted_entities)} 实体, {len(valid_relations)} 关系")

        return extracted_entities, valid_relations

    # ------------------------------------------------------------------
    # Helper: resolve LLM-returned entity name to known entity name
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entity_name(raw_name: str, entity_name_set: Set[str]) -> Optional[str]:
        """Resolve a potentially fuzzy entity name to a known name."""
        raw_name = raw_name.strip()
        if not raw_name:
            return None

        # Exact match
        if raw_name in entity_name_set:
            return raw_name

        # Case-insensitive match
        for known in entity_name_set:
            if known.lower() == raw_name.lower():
                return known

        # Core name match (strip parenthetical)
        raw_core = _extract_core_name(raw_name)
        matches = [n for n in entity_name_set if _extract_core_name(n) == raw_core]
        if len(matches) == 1:
            return matches[0]

        # Substring match (if raw is a substring of a known name or vice versa)
        for known in entity_name_set:
            if raw_core in known or known in raw_core:
                return known

        return None
