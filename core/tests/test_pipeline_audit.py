"""
Pipeline 审计优化后的 Prompt 验证测试。

验证内容：
1. RESOLVE_RELATION_PAIR_BATCH — 简化后的关系批量对齐
2. RESOLVE_ENTITY_CANDIDATES_BATCH — content 截断后的实体批量对齐
3. MERGE_MULTIPLE_ENTITY_CONTENTS — 内容合并质量
4. ENTITY_CONTENT_WRITE — 实体描述质量
5. RELATION_CONTENT_WRITE — 关系描述质量

运行：cd /home/linkco/exa/Deep-Dream && conda run -n base python core/tests/test_pipeline_audit.py
"""
import pytest
pytest.skip("Standalone script - run directly with python", allow_module_level=True)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import re
from core.llm.prompts import (
    RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
    MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT,
    ENTITY_CONTENT_WRITE_SYSTEM,
    ENTITY_CONTENT_WRITE_USER,
    RELATION_CONTENT_WRITE_SYSTEM,
    RELATION_CONTENT_WRITE_USER,
)
from core.llm.client import LLMClient

client = LLMClient(
    api_key="ollama",
    model_name="gemma4-26b-32k",
    base_url="http://localhost:11434/v1",
    context_window_tokens=32000,
    max_tokens=12000,
    think_mode=False,
)

def parse_json_block(text):
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    try:
        return json.loads(text.strip())
    except:
        return None

def call_llm(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client._call_llm("", system_prompt=system_prompt, messages=messages)
    return response

passed = 0
failed = 0
total = 0

def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# ============================================================
# 测试 1: RESOLVE_RELATION_PAIR_BATCH (简化后不含 merged_content)
# ============================================================
print("=" * 60)
print("测试 1: RESOLVE_RELATION_PAIR_BATCH (关系批量对齐)")
print("=" * 60)

cases = [
    ("匹配-相同关系", "A是B的父亲", ["A是B的生父"], "match_existing"),
    ("匹配-语义相同", "弓是弓矢的组成部分", ["弓矢由弓和箭组成"], "match_existing"),
    ("不匹配-不同行为", "在酒店休息避寒", ["在酒店喝酒写诗"], "create_new"),
    ("不匹配-不同关系", "曹操创建了魏国", ["曹操写了短歌行"], "create_new"),
]

for case_name, existing, new_rels, expected_action in cases:
    new_text = "\n".join(f"- 新关系{i+1}: {r}" for i, r in enumerate(new_rels))
    prompt = f"""<实体对>
- entity1: 概念A
- entity2: 概念B
</实体对>

<新关系描述>
{new_text}
</新关系描述>

<已有关系>
- family_id=rel_001 [source_document=test]: {existing}
</已有关系>"""

    resp = call_llm(RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT, prompt)
    result = parse_json_block(resp)
    if result:
        action = result.get("action", "")
        has_merged = "merged_content" in result and result.get("merged_content")
        test(f"{case_name}: action={action}", action == expected_action,
             f"expected={expected_action}, got={action}")
        test(f"{case_name}: 无冗余merged_content", not has_merged,
             f"LLM不应输出merged_content，got: {result}")
    else:
        test(f"{case_name}: JSON解析", False, f"parse failed: {resp[:100]}")


# ============================================================
# 测试 2: RESOLVE_ENTITY_CANDIDATES_BATCH (实体批量对齐)
# ============================================================
print("\n" + "=" * 60)
print("测试 2: RESOLVE_ENTITY_CANDIDATES_BATCH (实体批量对齐)")
print("=" * 60)

entity_cases = [
    ("别名合并", "士隐", "梦中见一僧一道",
     [{"family_id": "e1", "name": "甄士隐", "content": "姑苏望族，字士隐", "name_match_type": "substring"}],
     "merge_into_latest"),
    ("类型不同-创建新", "红楼梦", "曹雪芹创作的长篇小说",
     [{"family_id": "e2", "name": "曹雪芹", "content": "清代作家，名霑", "name_match_type": "none"}],
     "create_new"),
    ("同名合并", "贾宝玉", "衔玉而生",
     [{"family_id": "e3", "name": "贾宝玉", "content": "贾政之子", "name_match_type": "exact"}],
     ["reuse_existing", "merge_into_latest"]),
]

for case_name, name, content, candidates, expected_mode in entity_cases:
    cands_str = "\n\n".join(
        f"候选{i+1}:\n- family_id: {c['family_id']}\n- name: {c['name']}"
        + (f"\n- name_match_type: {c['name_match_type']}" if c.get('name_match_type') != 'none' else "")
        + f"\n- content: {c['content'][:200]}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""<当前实体>
- name: {name}
- content: {content}
</当前实体>

<候选实体列表>
{cands_str}
</候选实体列表>

请通过角色指纹对比判断对齐。"""

    resp = call_llm(RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT, prompt)
    result = parse_json_block(resp)
    if result:
        mode = result.get("update_mode", "")
        test(f"{case_name}: update_mode={mode}", mode in (expected_mode if isinstance(expected_mode, list) else [expected_mode]),
             f"expected={expected_mode}, got={mode}")
    else:
        test(f"{case_name}: JSON解析", False, f"parse failed: {resp[:100]}")


# ============================================================
# 测试 3: MERGE_MULTIPLE_ENTITY_CONTENTS (内容合并)
# ============================================================
print("\n" + "=" * 60)
print("测试 3: MERGE_MULTIPLE_ENTITY_CONTENTS (内容合并)")
print("=" * 60)

merge_cases = [
    ("新信息补充", "曹操，字孟德，东汉末年政治家", "曹操善用兵法，官渡之战击败袁绍",
     "应包含两段信息"),
    ("子集-无新信息", "刘备，字玄德，蜀汉开国皇帝，三国时期重要政治家", "刘备是蜀汉皇帝",
     "应返回旧版本原文"),
]

for case_name, old_content, new_content, check_desc in merge_cases:
    prompt = f"""<基础版本>
{old_content}
</基础版本>

<待融入的新信息>
新信息 1: {new_content}
</待融入的新信息>

在基础版本上做最小修改来融入新信息。禁止重写。无新信息则返回基础版本原文。直接输出合并后的文字，不要 JSON 包装。"""

    resp = call_llm(MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT, prompt)
    # Output is now plain text, not JSON
    merged = resp.strip()

    if case_name == "子集-无新信息":
        test(f"{case_name}: 保留旧版", old_content in merged or "刘备" in merged,
             f"旧内容应保留，got: {merged[:80]}")
    else:
        test(f"{case_name}: 包含新信息", "官渡" in merged or "袁绍" in merged,
             f"应包含新信息，got: {merged[:80]}")
        test(f"{case_name}: 包含旧信息", "孟德" in merged,
             f"应保留旧信息，got: {merged[:80]}")


# ============================================================
# 测试 4: ENTITY_CONTENT_WRITE (实体描述质量)
# ============================================================
print("\n" + "=" * 60)
print("测试 4: ENTITY_CONTENT_WRITE (实体描述质量)")
print("=" * 60)

text_sample = "甄士隐居住在姑苏城阊门外，家中虽不甚富贵，然本地也推他为望族。只因这甄士隐禀性恬淡，不以功名为念，每日只以观花修竹、酌酒吟诗为乐。"
user_prompt = ENTITY_CONTENT_WRITE_USER.format(entity_name="甄士隐", window_text=text_sample)
resp = call_llm(ENTITY_CONTENT_WRITE_SYSTEM, user_prompt)
result = parse_json_block(resp)
content = result.get("content", "") if result else ""

test("实体描述: 有内容", len(content) > 10, f"内容过短: '{content}'")
test("实体描述: 无模板开头", not content.startswith("该实体") and not content.startswith("这是一个"),
     f"有模板化开头: '{content[:20]}'")
test("实体描述: 长度合理", 30 <= len(content) <= 150, f"长度={len(content)}")


# ============================================================
# 测试 5: RELATION_CONTENT_WRITE (关系描述质量)
# ============================================================
print("\n" + "=" * 60)
print("测试 5: RELATION_CONTENT_WRITE (关系描述质量)")
print("=" * 60)

user_prompt = RELATION_CONTENT_WRITE_USER.format(
    entity_a="甄士隐", entity_b="贾雨村",
    window_text="甄士隐在家中宴请贾雨村，二人谈论诗文。雨村落魄时士隐曾赠银五十两助其赶考。"
)
resp = call_llm(RELATION_CONTENT_WRITE_SYSTEM, user_prompt)
result = parse_json_block(resp)
content = result.get("content", "") if result else ""

test("关系描述: 有内容", len(content) > 10, f"内容过短: '{content}'")
test("关系描述: 有具体关联", any(w in content for w in ["赠", "宴请", "资助", "谈论", "帮助"]),
     f"缺少具体关联: '{content}'")
test("关系描述: 长度合理", 10 <= len(content) <= 80, f"长度={len(content)}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print(f"测试结果: {passed}/{total} PASSED, {failed} FAILED")
print("=" * 60)
