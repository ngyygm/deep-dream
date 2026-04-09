"""LLM客户端 - 实体抽取相关操作。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..models import Episode
from ..utils import wprint
from .errors import LLMContextBudgetExceeded

# 多轮补充时仅追加本条 user，依赖首轮 system/user 与历史 assistant 中的任务与格式约定
_MULTI_ROUND_CONTINUE_USER = (
    "继续从文本中抽取概念实体。注意：\n"
    "1. 优先补充上一轮遗漏的：抽象概念、事件/过程、时间概念、描述性概念、文本锚点（章节标题等）\n"
    "2. 检查是否遗漏了文本中提到但尚未编目的关键概念\n"
    "3. 每个实体 content 应包含该概念在文中的具体行为、属性或角色"
)
from .prompts import (
    EXTRACT_ENTITIES_AND_RELATIONS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_RELATIONS_STRICT_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT,
    ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT,
    ENHANCE_ENTITY_JSON_RETRY_USER,
    DETAILED_JUDGMENT_PROCESS,
)


def _strip_yamlish_scalar(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _try_parse_bullet_name_content_entities(text: str) -> Optional[List[Dict[str, str]]]:
    """
    兼容模型误输出的 YAML 风格：交替的「- name: …」与「- content: …」行（非 JSON）。
    若无法识别为该类格式则返回 None。
    """
    raw = (text or "").strip()
    if not raw.startswith("-"):
        return None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("- name:"):
            i += 1
            continue
        name_val = _strip_yamlish_scalar(line[len("- name:") :].strip())
        if i + 1 >= len(lines):
            break
        line2 = lines[i + 1]
        if not line2.startswith("- content:"):
            i += 1
            continue
        content_val = _strip_yamlish_scalar(line2[len("- content:") :].strip())
        if name_val:
            out.append({"name": name_val, "content": content_val})
        i += 2
    return out if out else None


def _json_code_block(payload: Any) -> str:
    """将验收后的 JSON 结果重新包装为单个 json 代码块，供后续轮次复用。"""
    return f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"


class _EntityExtractionMixin:
    """实体抽取相关的 LLM 操作（mixin，通过 LLMClient 多继承使用）。"""

    def _parse_entities_list_from_response(self, response: str) -> List[Dict[str, str]]:
        """从 LLM 响应解析实体列表；JSON 非法时抛出 json.JSONDecodeError（供重试逻辑使用）。"""
        stripped = (response or "").strip()
        if stripped.startswith("-"):
            bullet = _try_parse_bullet_name_content_entities(response)
            if bullet is not None:
                entities: Any = bullet
            else:
                entities = self._parse_json_response(response)
        else:
            try:
                entities = self._parse_json_response(response)
            except json.JSONDecodeError:
                bullet = _try_parse_bullet_name_content_entities(response)
                if bullet is not None:
                    entities = bullet
                else:
                    raise
        if not isinstance(entities, list):
            entities = [entities]
        cleaned: List[Dict[str, str]] = []
        for entity in entities:
            if isinstance(entity, dict) and "name" in entity and "content" in entity:
                cleaned.append(
                    {
                        "name": str(entity["name"]).strip(),
                        "content": str(entity["content"]).strip(),
                    }
                )
        return cleaned

    def extract_entities_and_relations(self, episode: Episode, input_text: str,
                                        rounds: int = 1, verbose: bool = False,
                                        strict: bool = True,
                                        on_round_done=None) -> tuple:
        """
        合并抽取概念实体和实体间关系，支持多轮次补充抽取。

        多轮次利用 LLM 对话历史：第 2+ 轮把上一轮的 LLM 响应作为 assistant 消息，
        再追加一条 user 消息要求继续补充，天然携带已生成内容的完整上下文。

        Args:
            episode: 记忆缓存
            input_text: 输入文本
            rounds: 抽取轮次（默认 1）；>1 时每轮要求 LLM 继续补充
            verbose: 是否输出详细信息
            strict: 是否使用严格版 prompt（默认 True，禁止抽取抽象概念/类别词）
            on_round_done: 每轮完成后的回调 fn(round_idx, total_rounds, entity_count, rel_count)

        Returns:
            (entities, relations) — entities: List[Dict{name, content}],
                                    relations: List[Dict{entity1_name, entity2_name, content}]
        """
        system_prompt = EXTRACT_ENTITIES_RELATIONS_STRICT_SYSTEM_PROMPT if strict else EXTRACT_ENTITIES_AND_RELATIONS_SYSTEM_PROMPT
        prompt_episode = self._prepare_episode_for_prompt(episode)

        first_prompt = f"""<记忆缓存>
{prompt_episode}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

请从文本中同时抽取所有概念实体和实体间关系（越多越好）："""

        all_entities: List[Dict[str, str]] = []
        all_relations: List[Dict[str, str]] = []
        entity_name_to_index: Dict[str, int] = {}
        seen_rel_keys: set = set()

        # 构建对话历史
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]

        for round_idx in range(max(1, rounds)):
            if verbose:
                r, t = round_idx + 1, rounds
                wprint(f"【合并】轮{r}/{t}｜进行｜")

            try:
                (new_entities, new_relations), response = self.call_llm_until_json_parses(
                    messages,
                    parse_fn=lambda r: self._parse_merged_extraction_response(r),
                    json_parse_retries=2,
                )
            except LLMContextBudgetExceeded:
                if all_entities or all_relations:
                    wprint(
                        f"【合并】轮{round_idx + 1}/{rounds}｜上下文预算超限｜"
                        f"沿用已得 {len(all_entities)} 实体、{len(all_relations)} 关系，不再续轮"
                    )
                    break
                raise

            # 按名称去重合并实体；同名时保留 content 更长的版本
            accepted_entities: List[Dict[str, str]] = []
            accepted_entity_name_to_index: Dict[str, int] = {}
            new_entity_count = 0
            for e in new_entities:
                name = str(e.get("name") or "").strip()
                if not name:
                    continue
                candidate = {
                    "name": name,
                    "content": str(e.get("content") or "").strip(),
                }
                existing_idx = entity_name_to_index.get(name)
                if existing_idx is None:
                    entity_name_to_index[name] = len(all_entities)
                    all_entities.append(candidate)
                    accepted_entity_name_to_index[name] = len(accepted_entities)
                    accepted_entities.append(candidate)
                    new_entity_count += 1
                    continue
                if len(candidate["content"]) > len(all_entities[existing_idx]["content"]):
                    all_entities[existing_idx] = candidate
                    accepted_idx = accepted_entity_name_to_index.get(name)
                    if accepted_idx is None:
                        accepted_entity_name_to_index[name] = len(accepted_entities)
                        accepted_entities.append(candidate)
                    else:
                        accepted_entities[accepted_idx] = candidate

            # 按 (entity1, entity2) 去重合并关系
            accepted_relations: List[Dict[str, str]] = []
            new_rel_count = 0
            for r in new_relations:
                key = (r['entity1_name'], r['entity2_name'])
                if key not in seen_rel_keys:
                    all_relations.append(r)
                    seen_rel_keys.add(key)
                    accepted_relations.append(r)
                    new_rel_count += 1

            accepted_response = _json_code_block({
                "entities": accepted_entities,
                "relations": accepted_relations,
            })

            messages.append({"role": "assistant", "content": accepted_response})

            if verbose:
                r, t = round_idx + 1, rounds
                wprint(
                    f"【合并】轮{r}/{t}｜完成｜新{new_entity_count}实体 {new_rel_count}关系 "
                    f"累{len(all_entities)}实体 {len(all_relations)}关系"
                )

            if on_round_done:
                on_round_done(round_idx + 1, rounds, len(all_entities), len(all_relations))

            # 本轮无新增，提前退出
            if new_entity_count == 0 and new_rel_count == 0:
                if verbose:
                    r, t = round_idx + 1, rounds
                    wprint(f"【合并】轮{r}/{t}｜停止｜无新增")
                break

            # 还有下一轮：只追加简短续写指令，上下文已在首轮与历史 assistant 中
            if round_idx + 1 < rounds:
                if not self._can_continue_multi_round(
                    messages,
                    next_user_content=_MULTI_ROUND_CONTINUE_USER,
                    stage_label="步骤2(合并抽取)",
                ):
                    break
                messages.append({"role": "user", "content": _MULTI_ROUND_CONTINUE_USER})

        return all_entities, all_relations

    def _parse_merged_extraction_response(self, response: str) -> tuple:
        """
        解析合并抽取的 LLM 响应。

        Args:
            response: LLM 响应文本

        Returns:
            (entities, relations) — 解析后的实体和关系列表
        """
        try:
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                result = {"entities": [], "relations": []}

            # 解析实体
            raw_entities = result.get("entities", [])
            if not isinstance(raw_entities, list):
                raw_entities = [raw_entities]
            entities = []
            for e in raw_entities:
                if isinstance(e, dict) and 'name' in e and 'content' in e:
                    entities.append({
                        'name': str(e['name']).strip(),
                        'content': str(e['content']).strip()
                    })

            # 解析关系
            raw_relations = result.get("relations", [])
            if not isinstance(raw_relations, list):
                raw_relations = [raw_relations]
            relations = []
            for rel in raw_relations:
                if not isinstance(rel, dict):
                    continue
                entity1 = (rel.get('entity1_name') or '').strip()
                entity2 = (rel.get('entity2_name') or '').strip()
                content = (rel.get('content') or '').strip()
                if entity1 and entity2 and content:
                    # 标准化实体对（无向）
                    if entity1 > entity2:
                        entity1, entity2 = entity2, entity1
                    relations.append({
                        'entity1_name': entity1,
                        'entity2_name': entity2,
                        'content': content,
                    })

            # 过滤关系中引用了不存在实体的关系
            entity_names = {e['name'] for e in entities}
            valid_relations = []
            for rel in relations:
                if rel['entity1_name'] in entity_names and rel['entity2_name'] in entity_names:
                    valid_relations.append(rel)
                else:
                    # 尝试模糊匹配（LLM 可能简化了实体名称）
                    e1_matched = self._fuzzy_match_entity_name(rel['entity1_name'], entity_names)
                    e2_matched = self._fuzzy_match_entity_name(rel['entity2_name'], entity_names)
                    if e1_matched and e2_matched:
                        valid_relations.append({
                            'entity1_name': e1_matched,
                            'entity2_name': e2_matched,
                            'content': rel['content'],
                        })

            return entities, valid_relations

        except json.JSONDecodeError:
            raise
        except Exception as e:
            wprint(f"解析合并抽取JSON失败: {e}")
            wprint(f"响应内容: {response[:500]}...")
            return [], []

    @staticmethod
    def _fuzzy_match_entity_name(name: str, valid_names: set) -> Optional[str]:
        """尝试模糊匹配实体名称到有效名称集合。"""
        if name in valid_names:
            return name
        # 去除括号内容后匹配
        base_name = name.split('（')[0].split('(')[0].strip()
        for vn in valid_names:
            vn_base = vn.split('（')[0].split('(')[0].strip()
            if base_name == vn_base:
                return vn
        return None

    def extract_entities(self, episode: Episode, input_text: str,
                         rounds: int = 1, verbose: bool = False,
                         on_round_done=None,
                         compress_multi_round: bool = False) -> List[Dict[str, str]]:
        """
        抽取实体，支持多轮次补充抽取（利用 LLM 对话历史）。

        Args:
            episode: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            rounds: 抽取轮次（默认 1）；>1 时利用对话历史要求 LLM 继续补充
            verbose: 是否输出详细信息
            on_round_done: 每轮完成后的回调 fn(round_idx, total_rounds, cumulative_count)
            compress_multi_round: 兼容保留；多轮时与 False 相同：首轮完整 prompt，后续轮仅 user「继续生成」，依赖对话历史

        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        system_prompt = EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT
        prompt_episode = self._prepare_episode_for_prompt(episode)

        first_prompt = f"""<记忆缓存>
{prompt_episode}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

请从文本中抽取所有概念实体（越多越好）："""

        all_entities: List[Dict[str, str]] = []
        name_to_index: Dict[str, int] = {}

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
        # 压缩多轮：按轮追加 user/assistant，供蒸馏保存（无长 JSON 堆叠在单次请求外）
        distill_flat: List[Dict[str, str]] = []

        for round_idx in range(max(1, rounds)):
            if verbose:
                r, t = round_idx + 1, rounds
                wprint(f"【步骤2】轮{r}/{t}｜进行｜")

            try:
                new_entities, response = self.call_llm_until_json_parses(
                    messages,
                    parse_fn=lambda r: self._parse_entities_list_from_response(r),
                    json_parse_retries=4,
                )
            except LLMContextBudgetExceeded:
                if all_entities:
                    wprint(
                        f"【步骤2】轮{round_idx + 1}/{rounds}｜上下文预算超限｜"
                        f"沿用已得 {len(all_entities)} 个实体，不再续轮"
                    )
                    break
                raise

            # 验收并去重：同名实体只保留一个；若新版本 content 更长，则覆盖旧版本
            accepted_entities: List[Dict[str, str]] = []
            accepted_name_to_index: Dict[str, int] = {}
            new_count = 0
            for e in new_entities:
                name = str(e.get("name") or "").strip()
                if not name:
                    continue
                candidate = {
                    "name": name,
                    "content": str(e.get("content") or "").strip(),
                }
                existing_idx = name_to_index.get(name)
                if existing_idx is None:
                    name_to_index[name] = len(all_entities)
                    all_entities.append(candidate)
                    accepted_name_to_index[name] = len(accepted_entities)
                    accepted_entities.append(candidate)
                    new_count += 1
                    continue
                if len(candidate["content"]) > len(all_entities[existing_idx]["content"]):
                    all_entities[existing_idx] = candidate
                    accepted_idx = accepted_name_to_index.get(name)
                    if accepted_idx is None:
                        accepted_name_to_index[name] = len(accepted_entities)
                        accepted_entities.append(candidate)
                    else:
                        accepted_entities[accepted_idx] = candidate

            accepted_response = _json_code_block(accepted_entities)

            distill_flat.append({"role": "user", "content": messages[-1]["content"]})
            distill_flat.append({"role": "assistant", "content": accepted_response})

            messages.append({"role": "assistant", "content": accepted_response})

            if verbose:
                r, t = round_idx + 1, rounds
                wprint(
                    f"【步骤2】轮{r}/{t}｜完成｜新{new_count} 累{len(all_entities)}实体"
                )

            if on_round_done:
                on_round_done(round_idx + 1, rounds, len(all_entities))

            if new_count == 0:
                if verbose:
                    r, t = round_idx + 1, rounds
                    wprint(f"【步骤2】轮{r}/{t}｜停止｜无新增")
                break

            if round_idx + 1 < rounds:
                if not self._can_continue_multi_round(
                    messages,
                    next_user_content=_MULTI_ROUND_CONTINUE_USER,
                    stage_label="步骤2(实体抽取)",
                ):
                    break
                messages.append({"role": "user", "content": _MULTI_ROUND_CONTINUE_USER})

        if self._distill_data_dir and self._current_distill_step:
            if compress_multi_round:
                save_msgs = [{"role": "system", "content": system_prompt}] + distill_flat
            else:
                save_msgs = messages
            self._save_distill_conversation(save_msgs)

        return all_entities

    def extract_entities_by_names(self, episode: Episode, input_text: str,
                                  entity_names: List[str],
                                  verbose: bool = False) -> List[Dict[str, str]]:
        """
        抽取指定名称的实体

        Args:
            episode: 记忆缓存
            input_text: 输入文本
            entity_names: 要抽取的实体名称列表
            verbose: 是否输出详细信息

        Returns:
            抽取的实体列表，每个实体包含 name 和 content
        """
        if not entity_names:
            return []

        system_prompt = EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT
        prompt_episode = self._prepare_episode_for_prompt(episode)

        entity_names_str = "\n".join([f"- {name}" for name in entity_names])

        prompt = f"""<记忆缓存>
{prompt_episode}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

<指定实体名称>
{entity_names_str}
</指定实体名称>

请从输入文本中抽取上述指定的实体。只输出一个 ```json ... ``` 代码块；代码块内部必须是 JSON 数组，每个对象仅含 "name" 与 "content" 字符串字段；不要输出 YAML 或 `- name:` 格式。"""

        if verbose:
            wprint(f"【步骤4】补全｜调用｜{len(entity_names)}个名称")

        messages_sp = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        def _parse_and_filter(r: str) -> List[Dict[str, str]]:
            raw = self._parse_entities_list_from_response(r)
            out: List[Dict[str, str]] = []
            for e in raw:
                if e.get("name") in entity_names:
                    out.append(e)
            return out

        cleaned_entities, response = self.call_llm_until_json_parses(
            messages_sp,
            parse_fn=_parse_and_filter,
            json_parse_retries=2,
        )

        if verbose:
            wprint("【步骤4】补全｜解析｜")
            wprint(f"【步骤4】补全｜结果｜{len(cleaned_entities)}实体")

        return cleaned_entities

    def enhance_entity_content(self, episode: Episode, input_text: str,
                              entity: Dict[str, str]) -> str:
        """
        实体后验增强：结合缓存记忆和当前text对实体的content进行更细致的补全挖掘

        Args:
            episode: 当前的记忆缓存
            input_text: 当前窗口的输入文本
            entity: 实体字典，包含name和content

        Returns:
            增强后的实体content
        """
        system_prompt = ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT
        prompt_episode = self._prepare_episode_for_prompt(episode)

        prompt = f"""<记忆缓存>
{prompt_episode}
</记忆缓存>

<输入文本>
{input_text}
</输入文本>

<已抽取实体>
- 名称：{entity['name']}
- 当前content：{entity['content']}
</已抽取实体>

请对该实体的 content 进行增强。只输出与系统说明一致的 ```json ... ``` 代码块；代码块内部是单个 JSON 对象，且仅含键 "content"（值为增强后的完整描述字符串）。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        def _parse_enhance_response(response: str) -> str:
            result = self._parse_json_response(response)
            if not isinstance(result, dict) or "content" not in result:
                raise json.JSONDecodeError("enhance: 需要含 content 的 JSON 对象", response, 0)
            enhanced_content = str(result["content"]).strip()
            if not enhanced_content:
                raise json.JSONDecodeError("enhance: content 为空", response, 0)
            return enhanced_content

        try:
            enhanced, _ = self.call_llm_until_json_parses(
                messages,
                parse_fn=_parse_enhance_response,
                json_parse_retries=3,
                json_retry_user_message=ENHANCE_ENTITY_JSON_RETRY_USER,
            )
            return enhanced
        except (json.JSONDecodeError, Exception) as e:
            wprint(f"警告：实体后验增强 JSON 解析失败，保留原 content: {e}")
            return (entity.get("content") or "").strip()
