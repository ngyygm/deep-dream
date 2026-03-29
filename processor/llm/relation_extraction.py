"""LLM客户端 - 关系抽取相关操作。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..models import MemoryCache, Entity
from ..debug_log import log as dbg
from ..utils import wprint
from .errors import LLMContextBudgetExceeded
from .prompts import (
    EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT,
    JUDGE_NEED_CREATE_RELATION_SYSTEM_PROMPT,
    GENERATE_RELATION_CONTENT_SYSTEM_PROMPT,
)


def _json_code_block(payload: Any) -> str:
    """将验收后的 JSON 结果重新包装为单个 json 代码块，供后续轮次复用。"""
    return f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"


_MULTI_ROUND_CONTINUE_USER = "继续生成"


class _RelationExtractionMixin:
    """关系抽取相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    _RE_PAREN_FW = re.compile(r"（[^（）]*）")
    _RE_PAREN_ASCII = re.compile(r"\([^()]*\)")

    @classmethod
    def _strip_parenthetical_for_jaccard_compare(cls, name: str) -> str:
        """去掉全角/半角括号及其内文，仅用于 Jaccard 比较，不改变对外保留的规范名。"""
        s = (name or "").strip()
        if not s:
            return ""
        while True:
            prev = s
            s = cls._RE_PAREN_FW.sub("", s)
            s = cls._RE_PAREN_ASCII.sub("", s)
            if s == prev:
                break
        return s.strip()

    @staticmethod
    def _name_has_bracket_annotation(name: str) -> bool:
        """是否含可展示的括号说明（全角或半角成对）。"""
        s = name or ""
        return bool(_RelationExtractionMixin._RE_PAREN_FW.search(s)) or bool(
            _RelationExtractionMixin._RE_PAREN_ASCII.search(s)
        )

    def _build_relation_entity_catalog(
        self,
        entities: List[Dict[str, Any]],
        *,
        preferred_names: Optional[set] = None,
    ) -> tuple[str, Set[str], List[str]]:
        """构建关系列表字符串、合法名称集合、以及目录顺序（先见先排，用于并列时选较新名称）。

        relation_content_snippet_length <= 0 时目录中仅输出实体名（不附带 content 摘要）。
        """
        entity_lines: List[str] = []
        valid_names: Set[str] = set()
        name_order: List[str] = []
        order_seen: Set[str] = set()
        rc = int(self.relation_content_snippet_length or 0)
        name_only_catalog = rc <= 0
        snippet_limit = max(20, min(rc, 60)) if not name_only_catalog else 0

        ordered_entities = entities
        if preferred_names:
            preferred = []
            others = []
            for entity in entities:
                name = entity.get('name', '').strip()
                if name in preferred_names:
                    preferred.append(entity)
                else:
                    others.append(entity)
            ordered_entities = preferred + others

        for entity in ordered_entities:
            name = entity.get('name', '').strip()
            if not name:
                continue
            valid_names.add(name)
            if name not in order_seen:
                order_seen.add(name)
                name_order.append(name)
            if name_only_catalog:
                entity_lines.append(f"- {name}")
                continue
            content = entity.get('content', '').strip()
            if content:
                snippet = content[:snippet_limit] + ('...' if len(content) > snippet_limit else '')
                entity_lines.append(f"- {name} | {snippet}")
            else:
                entity_lines.append(f"- {name}")

        return '\n'.join(entity_lines), valid_names, name_order

    @staticmethod
    def _extract_relation_entity_id_token(entity_ref: Any) -> Optional[str]:
        """从 `E1 | 实体名` / `E1: 实体名` / `E1` 中提取规范编号。"""
        if entity_ref is None:
            return None
        ref = str(entity_ref).strip()
        if not ref:
            return None
        if ref.isdigit():
            return f"E{ref}"
        match = re.search(r"(?i)\bE\s*(\d+)\b", ref)
        if match:
            return f"E{match.group(1)}"
        return None

    @staticmethod
    def _extract_relation_entity_name_hint(entity_ref: Any) -> Optional[str]:
        """从 `E1 | 实体名` 这类混合字段中提取名称提示。"""
        if entity_ref is None:
            return None
        ref = str(entity_ref).strip()
        if not ref:
            return None
        token = _RelationExtractionMixin._extract_relation_entity_id_token(ref)
        if not token:
            return None
        token_match = re.search(r"(?i)\bE\s*\d+\b", ref)
        if not token_match:
            return None
        hint = ref[token_match.end():].strip()
        hint = hint.lstrip(" |:：-_,，;；/\\()（）[]【】")
        hint = hint.strip()
        return hint or None

    @staticmethod
    def _char_jaccard_similarity(a: str, b: str) -> float:
        """字符集合 Jaccard：|A∩B|/|A∪B|，用于关系端点与已知实体名的简单模糊匹配。"""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        u = sa | sb
        if not u:
            return 1.0
        return len(sa & sb) / len(u)

    def _tie_break_jaccard_entity_candidates(
        self,
        candidates: List[str],
        name_order: Optional[List[str]],
    ) -> str:
        """同 Jaccard 分：优先带括号说明的全名；若均带括号则取 catalog 中更靠后（新）的项。"""
        if len(candidates) == 1:
            return candidates[0]
        with_ann = [c for c in candidates if self._name_has_bracket_annotation(c)]
        pool = with_ann if with_ann else candidates
        if len(pool) == 1:
            return pool[0]
        if name_order:
            idx = {n: i for i, n in enumerate(name_order)}

            def sort_key(c: str) -> tuple:
                return (idx.get(c, -1), len(c), c)

            return max(pool, key=sort_key)
        return max(pool, key=lambda c: (len(c), c))

    def _best_jaccard_entity_match(
        self,
        name: str,
        valid_entity_names: set,
        threshold: float,
        *,
        name_order: Optional[List[str]] = None,
    ) -> Optional[str]:
        """对去掉括号后的字符串做 Jaccard；≥threshold 则映射回目录中的展示名（并列按括号与新旧规则）。"""
        name = (name or "").strip()
        if not name or not valid_entity_names:
            return None
        s_name = self._strip_parenthetical_for_jaccard_compare(name)
        if not s_name:
            s_name = name
        best_score = -1.0
        best: List[str] = []
        for vn in valid_entity_names:
            s_vn = self._strip_parenthetical_for_jaccard_compare(vn)
            if not s_vn:
                s_vn = vn.strip()
            j = self._char_jaccard_similarity(s_name, s_vn)
            if j > best_score + 1e-12:
                best_score = j
                best = [vn]
            elif abs(j - best_score) <= 1e-12:
                best.append(vn)
        if best_score < threshold:
            return None
        return self._tie_break_jaccard_entity_candidates(best, name_order)

    def _relation_endpoint_jaccard_threshold(self) -> float:
        return float(getattr(self, "relation_endpoint_jaccard_threshold", 0.9))

    def _relation_endpoint_embedding_threshold_value(self) -> Optional[float]:
        return getattr(self, "relation_endpoint_embedding_threshold", None)

    @staticmethod
    def _cosine_sim_queries_vs_catalog(q: np.ndarray, c: np.ndarray) -> np.ndarray:
        """q: (K,d), c: (M,d) → (K,M) 余弦相似度。"""
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        return qn @ cn.T

    def _build_endpoint_resolution_map_for_parse(
        self,
        unique_raws: List[str],
        known: Set[str],
        catalog_name_order: Optional[List[str]],
        catalog_emb: Optional[np.ndarray],
    ) -> Dict[str, str]:
        """
        端点解析：精确 / 基础名归一 / Jaccard；仍未知则对「去括号文本」批量编码，
        与目录向量一次性算余弦，≥阈值取 argmax（并列再按括号与 catalog 顺序）。
        """
        thr_j = self._relation_endpoint_jaccard_threshold()
        thr_e = self._relation_endpoint_embedding_threshold_value()
        ec = getattr(self, "_relation_embedding_client", None)
        order = catalog_name_order or []
        resolution: Dict[str, str] = {}
        pending_emb: List[str] = []

        for raw in unique_raws:
            if raw in known:
                resolution[raw] = raw
                continue
            n = self._normalize_entity_name_to_original(raw, known)
            if n in known:
                resolution[raw] = n
                continue
            jm = self._best_jaccard_entity_match(
                raw, known, thr_j, name_order=catalog_name_order
            )
            if jm:
                resolution[raw] = jm
                continue
            pending_emb.append(raw)

        if not pending_emb:
            return resolution

        use_vec = (
            catalog_emb is not None
            and thr_e is not None
            and ec is not None
            and getattr(ec, "is_available", lambda: False)()
            and order
            and catalog_emb.ndim == 2
            and catalog_emb.shape[0] == len(order)
        )
        if not use_vec:
            for ur in pending_emb:
                resolution[ur] = ur
            return resolution

        texts = [self._strip_parenthetical_for_jaccard_compare(ur) or ur for ur in pending_emb]
        qemb = ec.encode(texts)
        if qemb is None:
            for ur in pending_emb:
                resolution[ur] = ur
            return resolution

        q = np.asarray(qemb, dtype=np.float64)
        c = catalog_emb
        sims = self._cosine_sim_queries_vs_catalog(q, c)
        thr = float(thr_e)
        for i, ur in enumerate(pending_emb):
            row = sims[i]
            j_max = int(np.argmax(row))
            best_sim = float(row[j_max])
            if best_sim < thr:
                resolution[ur] = ur
                continue
            tie_mask = np.abs(row - best_sim) < 1e-6
            idxs = np.flatnonzero(tie_mask)
            candidates = [order[int(j)] for j in idxs]
            resolution[ur] = self._tie_break_jaccard_entity_candidates(
                candidates, catalog_name_order
            )
        return resolution

    def _relation_raw_pair_from_rel_dict(self, rel: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """从单条关系 dict 解析 entity1/entity2 原始名称；失败返回 None。"""
        raw1: Optional[str] = None
        raw2: Optional[str] = None
        if "entity1_name" in rel and "entity2_name" in rel:
            raw1 = str(rel["entity1_name"]).strip()
            raw2 = str(rel["entity2_name"]).strip()
        elif "实体1" in rel and "实体2" in rel:
            raw1 = str(rel["实体1"]).strip()
            raw2 = str(rel["实体2"]).strip()
        elif "实体A" in rel and "实体B" in rel:
            raw1 = str(rel["实体A"]).strip()
            raw2 = str(rel["实体B"]).strip()
        elif "from" in rel and "to" in rel:
            raw1 = str(rel["from"]).strip()
            raw2 = str(rel["to"]).strip()
        else:
            entity1_ref = (
                rel.get("entity1_id")
                or rel.get("entity1_ref")
                or rel.get("实体1编号")
                or rel.get("实体A编号")
            )
            entity2_ref = (
                rel.get("entity2_id")
                or rel.get("entity2_ref")
                or rel.get("实体2编号")
                or rel.get("实体B编号")
            )
            if entity1_ref is not None and entity2_ref is not None:
                raw1 = (
                    (rel.get("entity1_name") or rel.get("实体1名称") or rel.get("实体A名称"))
                    or self._extract_relation_entity_name_hint(entity1_ref)
                    or str(entity1_ref).strip()
                )
                raw2 = (
                    (rel.get("entity2_name") or rel.get("实体2名称") or rel.get("实体B名称"))
                    or self._extract_relation_entity_name_hint(entity2_ref)
                    or str(entity2_ref).strip()
                )
                if raw1:
                    raw1 = str(raw1).strip()
                if raw2:
                    raw2 = str(raw2).strip()
        if not raw1 or not raw2:
            return None
        return raw1, raw2

    def _resolve_relation_entity_name(
        self,
        provided_name: Optional[str],
        valid_entity_names: set,
        *,
        catalog_name_order: Optional[List[str]] = None,
    ) -> Optional[str]:
        """精确/基础名唯一映射；否则按 Jaccard≥阈值映射到唯一已知名，达不到则返回 None（由调用方保留原文）。"""
        name = (provided_name or "").strip()
        if not name or not valid_entity_names:
            return None
        if name in valid_entity_names:
            return name

        normalized = self._normalize_entity_name_to_original(name, valid_entity_names)
        if normalized in valid_entity_names:
            return normalized

        return self._best_jaccard_entity_match(
            name,
            valid_entity_names,
            self._relation_endpoint_jaccard_threshold(),
            name_order=catalog_name_order,
        )

    def _resolve_relation_endpoint_for_extraction(
        self,
        raw: Optional[str],
        valid_entity_names: set,
        *,
        catalog_name_order: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        解析关系端点：映射到列表内规范名；无法达到 Jaccard 阈值时保留 LLM 原文供步骤4补全。
        不再因「多候选分差过小」整条丢弃关系。
        """
        name = (raw or "").strip()
        if not name:
            return None
        if not valid_entity_names:
            return name
        if name in valid_entity_names:
            return name

        norm = self._normalize_entity_name_to_original(name, valid_entity_names)
        if norm in valid_entity_names:
            return norm

        matched = self._best_jaccard_entity_match(
            name,
            valid_entity_names,
            self._relation_endpoint_jaccard_threshold(),
            name_order=catalog_name_order,
        )
        if matched:
            return matched
        return name

    def _normalize_and_filter_relations_by_entities(
        self,
        relations: List[Dict[str, str]],
        known_entity_names: set,
        *,
        verbose: bool = False,
        catalog_name_order: Optional[List[str]] = None,
    ) -> tuple[List[Dict[str, str]], int, int]:
        """
        规范化关系端点：尽量映射到本窗已知实体名；未知名称保留供步骤4补全。
        丢弃缺端点、同端点、无内容的关系。
        """
        normalized_relations: List[Dict[str, str]] = []
        normalized_count = 0
        filtered_count = 0
        filtered_examples: List[str] = []

        for rel in relations:
            original_e1 = str(rel.get('entity1_name', '')).strip()
            original_e2 = str(rel.get('entity2_name', '')).strip()
            content = str(rel.get('content', '')).strip()

            if not original_e1 or not original_e2 or not content:
                filtered_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(f"{original_e1} <-> {original_e2}")
                continue
            if original_e1 == original_e2:
                filtered_count += 1
                if len(filtered_examples) < 5:
                    filtered_examples.append(f"{original_e1} <-> {original_e2}")
                continue

            n1 = self._normalize_entity_name_to_original(original_e1, known_entity_names)
            if n1 not in known_entity_names:
                n1 = self._resolve_relation_entity_name(
                    original_e1, known_entity_names, catalog_name_order=catalog_name_order
                ) or n1

            n2 = self._normalize_entity_name_to_original(original_e2, known_entity_names)
            if n2 not in known_entity_names:
                n2 = self._resolve_relation_entity_name(
                    original_e2, known_entity_names, catalog_name_order=catalog_name_order
                ) or n2

            if n1 != original_e1 or n2 != original_e2:
                normalized_count += 1

            rel_copy = rel.copy()
            rel_copy['entity1_name'] = n1
            rel_copy['entity2_name'] = n2
            rel_copy['content'] = content
            normalized_relations.append(rel_copy)

        if verbose and filtered_count > 0:
            suffix = f"，示例: {'; '.join(filtered_examples)}" if filtered_examples else ""
            wprint(f"【步骤3】过滤｜丢弃｜{filtered_count}条{suffix}")

        return normalized_relations, normalized_count, filtered_count

    def extract_relations(self, memory_cache: MemoryCache, input_text: str,
                         entities: Union[List[Dict[str, str]], List[Entity]],
                         rounds: int = 1,
                         verbose: bool = False,
                         on_round_done=None,
                         compress_multi_round: bool = False) -> List[Dict[str, str]]:
        """
        抽取概念关系边，支持多轮次补充抽取（利用 LLM 对话历史）。

        Args:
            memory_cache: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entities: 实体列表，可以是Dict（name, content）或Entity对象（包含entity_id）
            rounds: 抽取轮次（默认 1）；>1 时利用对话历史要求 LLM 继续补充
            verbose: 是否输出详细日志
            on_round_done: 每轮完成后的回调 fn(round_idx, total_rounds, cumulative_count)
            compress_multi_round: 兼容保留；多轮行为与 False 一致（首轮完整 user + 后续仅「继续生成」），为 True 时蒸馏保存仍用 system + distill_flat

        Returns:
            抽取的关系列表，每个关系包含 entity1_name, entity2_name, content
        """
        if not entities:
            return []

        # 统一转换为实体信息字典格式
        entity_info_list = []
        for entity in entities:
            if isinstance(entity, Entity):
                entity_info = {
                    'name': entity.name,
                    'content': entity.content,
                    'entity_id': entity.entity_id
                }
            else:
                entity_info = {
                    'name': entity['name'],
                    'content': entity['content'],
                    'entity_id': None
                }
            entity_info_list.append(entity_info)

        system_prompt = EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT

        entities_str, catalog_valid_names, catalog_name_order = self._build_relation_entity_catalog(
            entity_info_list
        )

        first_prompt = f"""<输入文本>
{input_text}
</输入文本>

<概念实体列表>
{entities_str}
</概念实体列表>

请从文本中抽取所有概念实体间的关系（越多越好）："""

        all_relations: List[Dict[str, str]] = []
        seen_rel_keys: set = set()  # (entity1, entity2, content_hash)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
        distill_flat: List[Dict[str, str]] = []

        # 收集所有有效实体名称，用于规范化
        valid_entity_names = {e.get('name', '').strip() for e in entity_info_list if e.get('name', '').strip()}

        def _accept_relations(candidates: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
            accepted_relations: List[Dict[str, str]] = []
            new_count = 0
            for rel in candidates:
                e1 = rel.get('entity1_name', '').strip()
                e2 = rel.get('entity2_name', '').strip()
                content = rel.get('content', '').strip()
                if not e1 or not e2 or not content:
                    continue
                # 标准化实体对
                if e1 > e2:
                    e1, e2 = e2, e1
                content_hash = hash(content.strip().lower())
                key = (e1, e2, content_hash)
                if key not in seen_rel_keys:
                    seen_rel_keys.add(key)
                    accepted_rel = {
                        'entity1_name': e1,
                        'entity2_name': e2,
                        'content': content,
                    }
                    all_relations.append(accepted_rel)
                    accepted_relations.append(accepted_rel)
                    new_count += 1
            return accepted_relations, new_count

        def _covered_entity_names(relations: List[Dict[str, str]]) -> set[str]:
            covered: set[str] = set()
            for rel in relations:
                e1 = rel.get('entity1_name', '').strip()
                e2 = rel.get('entity2_name', '').strip()
                if e1:
                    covered.add(e1)
                if e2:
                    covered.add(e2)
            return covered

        for round_idx in range(max(1, rounds)):
            if verbose:
                r, t = round_idx + 1, rounds
                wprint(f"【步骤3】轮{r}/{t}｜进行｜")

            try:
                new_relations, response = self.call_llm_until_json_parses(
                    messages,
                    parse_fn=lambda r, vs=catalog_valid_names, co=catalog_name_order: self._parse_relations_response(
                        r, vs, catalog_name_order=co
                    ),
                    json_parse_retries=2,
                )
            except LLMContextBudgetExceeded:
                if all_relations:
                    wprint(
                        f"【步骤3】轮{round_idx + 1}/{rounds}｜上下文预算超限｜"
                        f"沿用已得 {len(all_relations)} 条关系，不再续轮"
                    )
                    break
                raise
            dbg(f"关系抽取第{round_idx+1}轮 LLM原始响应 ({len(response)} 字符): {response[:800]}")

            new_relations, _normalized_count, _filtered_count = self._normalize_and_filter_relations_by_entities(
                new_relations, valid_entity_names, verbose=verbose, catalog_name_order=catalog_name_order
            )

            # 验收并去重：仅将本轮真正新增且合法的关系作为“被系统接受”的 assistant 输出
            accepted_relations, new_count = _accept_relations(new_relations)

            accepted_response = _json_code_block(accepted_relations)

            distill_flat.append({"role": "user", "content": messages[-1]["content"]})
            distill_flat.append({"role": "assistant", "content": accepted_response})

            messages.append({"role": "assistant", "content": accepted_response})

            if verbose:
                r, t = round_idx + 1, rounds
                wprint(
                    f"【步骤3】轮{r}/{t}｜完成｜新{new_count} 累{len(all_relations)}关系"
                )

            if on_round_done:
                on_round_done(round_idx + 1, rounds, len(all_relations))

            if new_count == 0:
                if verbose:
                    r, t = round_idx + 1, rounds
                    wprint(f"【步骤3】轮{r}/{t}｜停止｜无新增")
                break

            if round_idx + 1 < rounds:
                if not self._can_continue_multi_round(
                    messages,
                    next_user_content=_MULTI_ROUND_CONTINUE_USER,
                    stage_label="步骤3(关系抽取)",
                ):
                    covered_names = _covered_entity_names(all_relations)
                    uncovered_names = [
                        name for name in catalog_name_order
                        if name in valid_entity_names and name not in covered_names
                    ]
                    if uncovered_names:
                        if verbose:
                            wprint(
                                f"【步骤3】轮{round_idx + 2}/{rounds}｜退化补抽｜"
                                f"续轮上下文过长，改为单轮补抽未覆盖实体 {len(uncovered_names)} 个"
                            )
                        try:
                            fallback_relations = self._extract_relations_single_pass(
                                memory_cache,
                                input_text,
                                entity_info_list,
                                existing_relations=None,
                                uncovered_entities=uncovered_names,
                                verbose=verbose,
                            )
                        except LLMContextBudgetExceeded:
                            if verbose:
                                wprint(
                                    f"【步骤3】轮{round_idx + 2}/{rounds}｜退化补抽失败｜"
                                    "单轮补抽仍超出上下文预算，沿用当前关系结果"
                                )
                            break
                        fallback_accepted, fallback_new_count = _accept_relations(fallback_relations)
                        if verbose:
                            wprint(
                                f"【步骤3】轮{round_idx + 2}/{rounds}｜退化补抽完成｜"
                                f"新{fallback_new_count} 累{len(all_relations)}关系"
                            )
                        if on_round_done:
                            on_round_done(round_idx + 2, rounds, len(all_relations))
                        accepted_response = _json_code_block(fallback_accepted)
                        distill_flat.append({"role": "user", "content": first_prompt})
                        distill_flat.append({"role": "assistant", "content": accepted_response})
                    break
                messages.append({"role": "user", "content": _MULTI_ROUND_CONTINUE_USER})

        if self._distill_data_dir and self._current_distill_step:
            if compress_multi_round:
                save_msgs = [{"role": "system", "content": system_prompt}] + distill_flat
            else:
                save_msgs = messages
            self._save_distill_conversation(save_msgs)

        if verbose:
            wprint(
                f"【步骤3】汇总｜得｜{len(all_relations)}关系 实体{len(entity_info_list)} 轮{max(1, rounds)}"
            )
            if len(all_relations) == 0 and len(entity_info_list) > 1:
                wprint(
                    f"【步骤3】警告｜空关系｜{len(entity_info_list)}实体无关系（可能JSON或文本原因）"
                )
        return all_relations

    def _extract_relations_single_pass(self, memory_cache: MemoryCache,
                                      input_text: str,
                                      entities: List[Dict[str, Any]],
                                      existing_relations: Optional[Dict[tuple, List[str]]] = None,
                                      uncovered_entities: Optional[List[str]] = None,
                                      verbose: bool = False) -> List[Dict[str, str]]:
        """
        单次关系抽取

        Args:
            memory_cache: 记忆缓存
            input_text: 输入文本
            entities: 实体信息列表（包含name, content, entity_id）
            existing_relations: 已抽取的关系字典，key是(entity1, entity2)，value是关系content列表
            uncovered_entities: 未覆盖的实体名称列表（还没有任何关系，需要优先补全）
            verbose: 是否输出详细日志
        """
        system_prompt = EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT

        # 构建实体名称集合，用于区分已覆盖和未覆盖的实体
        uncovered_set = set(uncovered_entities) if uncovered_entities else set()
        entities_str, catalog_valid_names, catalog_name_order = self._build_relation_entity_catalog(
            entities,
            preferred_names=uncovered_set if uncovered_set else None,
        )
        preferred_uncovered_names = sorted(
            n for n in uncovered_set if n in catalog_valid_names
        )

        # 构建 prompt：顺序固定为 <输入文本> → <概念实体列表> → <已有关系> → <未覆盖实体>
        prompt_parts = []

        prompt_parts.append(f"<输入文本>\n{input_text}\n</输入文本>")

        prompt_parts.append(f"<概念实体列表>\n{entities_str}\n</概念实体列表>")

        if existing_relations:
            relations_info = []
            for (e1, e2), contents in existing_relations.items():
                if contents:
                    relations_info.append(f"- {e1} ↔ {e2}：{' / '.join(contents)}")
            if relations_info:
                prompt_parts.append(f"<已有关系>\n" + "\n".join(relations_info) + "\n</已有关系>")

        if preferred_uncovered_names:
            prompt_parts.append(
                "<未覆盖实体>\n\n"
                + "\n".join(f"- {n}" for n in preferred_uncovered_names)
                + "\n\n</未覆盖实体>\n\n"
                + "【注意】请优先为上述尚未出现在任何已抽取关系中的实体补全关系边；"
                "输出仅使用 entity1_name / entity2_name / content，不要输出 ID 或编号；只输出一个 ```json ... ``` 代码块。"
            )

        # 简洁的任务说明
        task_instruction = "请抽取概念实体间的关系。"
        if existing_relations:
            task_instruction += "不要重复已有关系。"
        prompt_parts.append(task_instruction)

        prompt = "\n\n".join(prompt_parts)

        if verbose:
            wprint(f"【步骤3】单次｜调用｜{len(entities)}实体")
        messages_sp = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result, response = self.call_llm_until_json_parses(
            messages_sp,
            parse_fn=lambda r, vs=catalog_valid_names, co=catalog_name_order: self._parse_relations_response(
                r, vs, catalog_name_order=co
            ),
            json_parse_retries=2,
        )
        if verbose:
            wprint("【步骤3】单次｜解析｜")

        valid_entity_names = {e.get('name', '').strip() for e in entities if e.get('name', '').strip()}
        result, normalized_count, filtered_count = self._normalize_and_filter_relations_by_entities(
            result, valid_entity_names, verbose=verbose, catalog_name_order=catalog_name_order
        )

        if verbose:
            if normalized_count > 0 or filtered_count > 0:
                parts = []
                if normalized_count > 0:
                    parts.append(f"规范化了 {normalized_count} 个实体名称")
                if filtered_count > 0:
                    parts.append(f"过滤了 {filtered_count} 条非法关系")
                wprint(
                    f"【步骤3】单次｜结果｜{len(result)}关系（{'，'.join(parts)}）"
                )
            else:
                wprint(f"【步骤3】单次｜结果｜{len(result)}关系")
        return result

    def _deduplicate_relations(self, relations: List[Dict[str, str]],
                               seen_relations: set) -> List[Dict[str, str]]:
        """
        对关系进行去重（代码层面的去重规则）
        关系是无向的，将(A,B)和(B,A)视为同一个关系

        Args:
            relations: 关系列表
            seen_relations: 已见过的关系集合，元素是 (entity1_name, entity2_name, content_hash)

        Returns:
            去重后的关系列表
        """
        deduplicated = []
        for rel in relations:
            entity1_name = rel.get('entity1_name', '').strip()
            entity2_name = rel.get('entity2_name', '').strip()
            content = rel.get('content', '').strip()

            if not entity1_name or not entity2_name or not content:
                continue

            # 标准化实体对（按字母顺序排序，使关系无向化）
            normalized_pair = self._normalize_entity_pair(entity1_name, entity2_name)
            normalized_entity1 = normalized_pair[0]
            normalized_entity2 = normalized_pair[1]

            # 使用关系内容的哈希值来去重（允许同一实体对存在多个不同关系）
            content_hash = hash(content.lower())
            # 使用标准化后的实体对作为key，使(A,B)和(B,A)被视为同一个关系
            relation_key = (normalized_entity1, normalized_entity2, content_hash)

            # 如果这个关系还没有见过，添加到结果中
            if relation_key not in seen_relations:
                # 确保输出的关系使用标准化后的实体对
                rel_copy = rel.copy()
                rel_copy['entity1_name'] = normalized_entity1
                rel_copy['entity2_name'] = normalized_entity2
                deduplicated.append(rel_copy)

        return deduplicated

    def _parse_relations_response(
        self,
        response: str,
        valid_entity_names: Optional[Set[str]] = None,
        *,
        catalog_name_order: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        解析关系抽取的LLM响应（纯名称主链：仅 entity1_name / entity2_name / content）。

        Args:
            response: LLM的响应文本
            valid_entity_names: 本窗步骤2 实体名称集合；Jaccard 在去掉括号说明后计算；
                仍不匹配且配置了 embedding 时，对目录名与待解析端点批量向量、余弦一次性比对。
            catalog_name_order: 与构建概念列表一致的名称顺序，供并列打破。

        Returns:
            解析后的关系列表
        """
        try:
            relations = self._parse_json_response(response)
            # 确保返回的是列表
            if not isinstance(relations, list):
                relations = [relations]

            known = set(valid_entity_names or ())
            co = catalog_name_order or []

            parsed_rows: List[Tuple[str, str, Dict[str, Any]]] = []
            unique_raw_list: List[str] = []
            seen_raw: Set[str] = set()

            for rel in relations:
                if not isinstance(rel, dict):
                    wprint(f"【步骤3】警告｜跳过｜格式无效 {rel}")
                    continue
                pair = self._relation_raw_pair_from_rel_dict(rel)
                if not pair:
                    wprint(f"【步骤3】警告｜跳过｜缺端点名 {rel}")
                    continue
                raw1, raw2 = pair
                parsed_rows.append((raw1, raw2, rel))
                for r in (raw1, raw2):
                    if r not in seen_raw:
                        seen_raw.add(r)
                        unique_raw_list.append(r)

            catalog_emb: Optional[np.ndarray] = None
            thr_e = self._relation_endpoint_embedding_threshold_value()
            ec = getattr(self, "_relation_embedding_client", None)
            if co and thr_e is not None and ec is not None and ec.is_available():
                emb = ec.encode(co)
                if emb is not None:
                    catalog_emb = np.asarray(emb, dtype=np.float64)

            resolution_map = self._build_endpoint_resolution_map_for_parse(
                unique_raw_list, known, catalog_name_order, catalog_emb
            )

            valid_relations: List[Dict[str, str]] = []
            for raw1, raw2, rel in parsed_rows:
                entity1 = resolution_map.get(raw1, raw1)
                entity2 = resolution_map.get(raw2, raw2)

                if not entity1 or not entity2:
                    wprint(f"【步骤3】警告｜跳过｜端点空 {rel}")
                    continue

                # 标准化实体对（按字母顺序排序，使关系无向化）
                normalized_pair = self._normalize_entity_pair(entity1, entity2)

                # 获取content（支持英文和中文键名）
                content = ''
                if 'content' in rel:
                    content = str(rel['content']).strip()
                elif '内容' in rel:
                    content = str(rel['内容']).strip()
                elif '关系内容' in rel:
                    content = str(rel['关系内容']).strip()
                elif '描述' in rel:
                    content = str(rel['描述']).strip()

                # 只保留必需的字段，移除其他字段（如relation_id）
                cleaned_relation = {
                    'entity1_name': normalized_pair[0],  # 使用标准化后的顺序
                    'entity2_name': normalized_pair[1],
                    'content': content
                }
                valid_relations.append(cleaned_relation)
            wprint(
                f"【步骤3】解析｜统计｜返{len(relations)} 有效{len(valid_relations)}"
            )
            dbg(f"关系解析: LLM返回 {len(relations)} 条, 有效 {len(valid_relations)} 条")
            if len(relations) != len(valid_relations):
                dbg(f"  无效关系详情:")
                for _rel in relations:
                    if not isinstance(_rel, dict):
                        dbg(f"    非dict类型: {_rel}")
                        continue
                    _e1 = (
                        _rel.get('entity1_id') or _rel.get('entity1_ref')
                        or _rel.get('entity1_name') or _rel.get('实体1编号')
                        or _rel.get('实体1') or _rel.get('实体A')
                    )
                    _e2 = (
                        _rel.get('entity2_id') or _rel.get('entity2_ref')
                        or _rel.get('entity2_name') or _rel.get('实体2编号')
                        or _rel.get('实体2') or _rel.get('实体B')
                    )
                    if not _e1 or not _e2:
                        dbg(f"    缺少实体名称: {_rel}")
            return valid_relations
        except json.JSONDecodeError:
            # 交给上层 call_llm_until_json_parses 重试 LLM
            raise
        except Exception as e:
            wprint(f"【步骤3】解析｜失败｜{e}")
            wprint(f"【步骤3】解析｜片段｜{response[:500]}")
            dbg(f"关系解析失败: {e}")
            dbg(f"  LLM响应前800字符: {response[:800]}")
            return []

    def judge_need_create_relation(self, entity1_name: str, entity1_content: str,
                                    entity2_name: str, entity2_content: str,
                                    entity1_memory_cache: Optional[str] = None,
                                    entity2_memory_cache: Optional[str] = None) -> bool:
        """
        判断两个实体之间是否真的需要创建关系边

        这个方法在生成关系content之前调用，使用两个实体的完整信息（name、content、memory_cache）
        来判断是否确实存在明确的、有意义的关联。

        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            entity1_memory_cache: 起始实体的记忆缓存内容（可选）
            entity2_memory_cache: 目标实体的记忆缓存内容（可选）

        Returns:
            True表示确实需要创建关系边，False表示不需要
        """
        system_prompt = JUDGE_NEED_CREATE_RELATION_SYSTEM_PROMPT

        # 构建实体信息
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content}
"""
        if entity1_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity1_memory_cache[:500]}...\n"

        entities_info += f"""
目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content}
"""
        if entity2_memory_cache:
            entities_info += f"- 相关记忆缓存: {entity2_memory_cache[:500]}...\n"

        prompt = f"""<实体信息>
{entities_info}
</实体信息>

请判断以下两个实体之间是否需要创建关系边。

只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"need_create": true/false}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            if isinstance(result, dict) and "need_create" in result:
                return bool(result["need_create"])
            else:
                # 如果解析失败，默认返回False（保守策略）
                return False

        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析判断关系JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")
            # 如果修复失败，默认返回False（保守策略）
            return False

    def generate_relation_content(self, entity1_name: str, entity1_content: str,
                                  entity2_name: str, entity2_content: str,
                                  relation_memory_cache: Optional[str] = None,
                                  preliminary_content: Optional[str] = None) -> str:
        """
        根据两个实体和memory_cache生成关系的content

        Args:
            entity1_name: 起始实体名称
            entity1_content: 起始实体内容描述
            entity2_name: 目标实体名称
            entity2_content: 目标实体内容描述
            relation_memory_cache: 关系的记忆缓存内容（可选，这是总结两个实体的memory_cache）
            preliminary_content: 初步的关系content（可选，用于参考）

        Returns:
            生成的关系content（自然语言描述）
        """
        system_prompt = GENERATE_RELATION_CONTENT_SYSTEM_PROMPT

        # 构建初步关系描述信息
        preliminary_info = ""
        if preliminary_content:
            preliminary_info = f"""
初步关系描述（作为参考）：
{preliminary_content}
"""

        # 构建实体信息（简化，只提供基本信息）
        entities_info = f"""
起始实体：
- 名称: {entity1_name}
- 描述: {entity1_content[:200]}{'...' if len(entity1_content) > 200 else ''}

目标实体：
- 名称: {entity2_name}
- 描述: {entity2_content[:200]}{'...' if len(entity2_content) > 200 else ''}
"""

        # 构建关系记忆缓存信息（只关注与关系相关的部分）
        relation_context = ""
        if relation_memory_cache:
            relation_context = f"""
关系记忆缓存：
{relation_memory_cache[:500]}{'...' if len(relation_memory_cache) > 500 else ''}
"""

        prompt = f"""<实体信息>
{preliminary_info}
{entities_info}
{relation_context}
</实体信息>

只输出一个 ```json ... ``` 代码块；代码块内部格式为：{{"content": "关系描述"}}"""

        response = self._call_llm(prompt, system_prompt)

        # 尝试解析JSON响应
        try:
            result = self._parse_json_response(response)

            if isinstance(result, dict) and "content" in result:
                return result["content"]
            else:
                # 如果解析失败，返回默认描述
                return f"'{entity1_name}'与'{entity2_name}'的关联关系"

        except (json.JSONDecodeError, Exception) as e:
            wprint(f"解析关系content JSON失败: {e}")
            # 显示更多调试信息
            error_pos = getattr(e, 'pos', None)
            if error_pos is not None:
                start = max(0, error_pos - 100)
                end = min(len(response), error_pos + 100)
                wprint(f"错误位置: {error_pos}, 附近内容: {response[start:end]}")
            else:
                wprint(f"响应内容前500字符: {response[:500]}...")

            # 尝试直接从响应中提取content字段的值（使用正则表达式）
            try:
                import re
                content_match = re.search(r'"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response)
                if content_match:
                    content_value = content_match.group(1)
                    content_value = content_value.encode().decode('unicode_escape')
                    wprint(f"使用正则表达式提取的content: {content_value[:100]}...")
                    return content_value
            except Exception as regex_error:
                wprint(f"正则表达式提取失败: {regex_error}")

            # 最后回退到默认描述
            wprint(f"所有修复尝试都失败，使用默认描述")
            return f"'{entity1_name}'与'{entity2_name}'的关联关系"
