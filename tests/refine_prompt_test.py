"""
Refinement Prompt A/B/C/D Comparison Test (Ollama Native API)
=============================================================
Uses Ollama native /api/chat endpoint (same as Deep-Dream server).

Variants:
  A) Original: "请检查是否有遗漏的关系..."
  B) Direct challenge: "你找的还不够全面..."
  C) Structured heuristics: 因果/时空/互动/亲属/情感 angles
  D) Concrete assertion: "至少还有5对遗漏"
  E) Hybrid: initial=false + refine=true

All variants do 2 refinement rounds on the same initial extraction result.
"""

import json
import time
import re
from urllib import request
from typing import Any, Dict, List, Optional

OLLAMA_BASE = "http://localhost:11434"
MODEL = "gemma4-26b-32k"
MAX_TOKENS = 16000
ROUNDS = 2

# --- Prompt templates ---

RELATION_DISCOVER_SYSTEM = """你是关系发现专家。从文本中找出概念之间人类会自然联想到的一切联系。
核心理念：任何两个概念在文本中有交互、关联或共现因果，都应发现。"""

RELATION_DISCOVER_USER = """给定概念列表，从文本中找出有人类可感知关联的概念对。

关联范围：任何人类会自然联想到的联系。宁多勿少。关系内容必须具体（一句话能说清），泛泛描述无效。

每个概念对只需出现一次（A→B 和 B→A 视为同一对）。

概念列表：{entity_names}

文本：
{window_text}

只输出一个```json```代码块，内部是概念对数组（每对只需出现一次）：
```json
[["概念A", "概念B"], ["概念C", "概念D"]]
```"""

# Variant A: Original (current)
REFINE_A = """请检查是否有遗漏的关系，特别关注之前未出现在任何关系对中的概念。如果没有，返回空数组。"""

# Variant B: Direct challenge (user's idea)
REFINE_B = """你找的关系还不够全面，还有很多遗漏。仔细再看一遍文本，把所有你之前没有列出的关系对都找出来。
特别关注：之前没有出现在任何关系对中的概念，以及你认为"不够重要"但确实有关联的概念对。
如果没有，返回空数组。"""

# Variant C: Structured heuristics
REFINE_C = """请从以下5个角度重新审视文本，找出遗漏的关系对：
1. 因果关系：一个事件导致另一个事件发生
2. 时空关系：同一时间或地点出现的概念
3. 互动关系：有直接对话、行为互动的概念
4. 社会关系：家庭、从属、身份关联
5. 对比/相似关系：两个概念形成对比或类比

每个角度都可能有遗漏。返回所有新发现的关系对。如果确实没有遗漏，返回空数组。"""

# Variant D: Concrete assertion
REFINE_D = """我仔细核对了文本，你至少遗漏了5对关系。请逐段重新阅读文本，找出所有你之前没有列出的关系对。
不要遗漏任何有关联的概念对。返回所有新发现的关系对。"""

REFINE_PROMPTS = {
    "A_original": REFINE_A,
    "B_challenge": REFINE_B,
    "C_structured": REFINE_C,
    "D_assertion": REFINE_D,
}


def ollama_chat(messages, think=False, num_predict=None):
    """Call Ollama native /api/chat (same as server)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": think,
    }
    if num_predict is not None:
        payload["num_predict"] = num_predict

    req = request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - t0

    message = data.get("message") or {}
    content = message.get("content", "") or ""
    thinking = message.get("thinking") or message.get("reasoning") or None
    done_reason = data.get("done_reason", "unknown")
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return content, thinking, elapsed, done_reason, prompt_tokens, completion_tokens


def parse_pairs(text):
    """Extract pairs from JSON response text."""
    # Try JSON block first
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            pairs = json.loads(json_match.group(1))
            if isinstance(pairs, list):
                return [tuple(sorted(p)) for p in pairs if isinstance(p, (list, tuple)) and len(p) == 2]
        except json.JSONDecodeError:
            pass
    # Direct JSON
    try:
        pairs = json.loads(text)
        if isinstance(pairs, list):
            return [tuple(sorted(p)) for p in pairs if isinstance(p, (list, tuple)) and len(p) == 2]
    except json.JSONDecodeError:
        pass
    # Pattern fallback
    pair_pattern = re.findall(r'\["([^"]+)",\s*"([^"]+)"\]', text)
    if pair_pattern:
        return [tuple(sorted((a, b))) for a, b in pair_pattern]
    return []


def normalize_pair(pair):
    return tuple(sorted(pair))


def run_variant(name, refine_prompt, initial_messages, initial_pairs, think_refine=False):
    """Run refinement rounds with given prompt and think mode."""
    messages = list(initial_messages)
    all_pairs = set(initial_pairs)
    total_new = 0
    total_time = 0

    print(f"\n{'='*60}")
    print(f"Variant: {name} (refine_think={think_refine})")
    print(f"{'='*60}")

    for r in range(ROUNDS):
        messages.append({"role": "user", "content": refine_prompt})
        content, thinking, elapsed, done_reason, ptok, ctok = ollama_chat(messages, think=think_refine)
        total_time += elapsed

        # Parse from content
        round_pairs = parse_pairs(content)
        source = "content"
        if not round_pairs and thinking:
            round_pairs = parse_pairs(thinking)
            source = "thinking"

        new_pairs = [p for p in round_pairs if normalize_pair(p) not in all_pairs]
        for p in new_pairs:
            all_pairs.add(normalize_pair(p))
        total_new += len(new_pairs)

        print(f"  R{r+1}: {elapsed:.1f}s, {ptok}+{ctok}tok, done={done_reason}, "
              f"{len(round_pairs)} from {source}, {len(new_pairs)} new → total {len(all_pairs)}")
        if new_pairs:
            for p in new_pairs:
                print(f"    + {p[0]} ↔ {p[1]}")

        # Add assistant response to conversation
        assistant_text = content if content else (thinking or "[]")
        messages.append({"role": "assistant", "content": assistant_text})

    return {
        "name": name,
        "initial": len(initial_pairs),
        "refine_new": total_new,
        "total": len(all_pairs),
        "time": total_time,
        "think": think_refine,
    }


def main():
    with open("/home/linkco/exa/Deep-Dream/core/tests/e2e_test_text.txt") as f:
        text = f.read()

    entity_names = [
        "甄士隐", "贾雨村", "封氏", "英莲", "霍启",
        "葫芦庙", "十里街", "好了歌", "跛足道人",
        "林如海", "贾敏", "林黛玉", "智通寺",
        "通灵宝玉", "女娲", "大荒山", "姑苏",
        "封肃", "神仙", "好了歌注",
    ]
    entity_list_str = "、".join(entity_names)

    # --- Phase 1: Initial extraction (think=false) ---
    print("=" * 60)
    print("PHASE 1: Initial extraction (think=false, native /api/chat)")
    print("=" * 60)

    initial_messages = [
        {"role": "system", "content": RELATION_DISCOVER_SYSTEM},
        {"role": "user", "content": RELATION_DISCOVER_USER.format(
            entity_names=entity_list_str, window_text=text
        )},
    ]

    content0, thinking0, elapsed0, done0, ptok0, ctok0 = ollama_chat(initial_messages, think=False)

    initial_pairs = parse_pairs(content0)
    parse_src = "content"
    if not initial_pairs and thinking0:
        initial_pairs = parse_pairs(thinking0)
        parse_src = "thinking"

    initial_set = set(normalize_pair(p) for p in initial_pairs)

    print(f"Time: {elapsed0:.1f}s, tokens: {ptok0}+{ctok0}, done={done0}")
    print(f"Content: {len(content0)} chars, Thinking: {len(thinking0 or '')} chars")
    print(f"Parsed {len(initial_pairs)} pairs from {parse_src}")
    if initial_pairs:
        for p in initial_pairs[:15]:
            print(f"  {p[0]} ↔ {p[1]}")
        if len(initial_pairs) > 15:
            print(f"  ... and {len(initial_pairs)-15} more")

    assistant_text = content0 if content0 else (thinking0 or "[]")
    initial_messages.append({"role": "assistant", "content": assistant_text})

    # --- Phase 2: Refinement variants ---
    print("\n" + "=" * 60)
    print("PHASE 2: Refinement variants comparison")
    print("=" * 60)

    results = []

    for name, prompt in REFINE_PROMPTS.items():
        result = run_variant(name, prompt, initial_messages, initial_set, think_refine=False)
        results.append(result)

    # Variant E: Hybrid (initial=false + refine=true with original prompt)
    result = run_variant("E_hybrid_t=true", REFINE_A, initial_messages, initial_set, think_refine=True)
    results.append(result)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<20} {'Init':>5} {'New':>5} {'Total':>6} {'Time':>8} {'Think':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<20} {r['initial']:>5} {r['refine_new']:>5} {r['total']:>6} {r['time']:>7.1f}s {str(r['think']):>6}")

    best = max(results, key=lambda r: r['total'])
    print(f"\nMost pairs: {best['name']} ({best['total']} pairs)")
    fastest = min(results, key=lambda r: r['time'])
    print(f"Fastest refine: {fastest['name']} ({fastest['time']:.1f}s)")


if __name__ == "__main__":
    main()
