#!/usr/bin/env python3
"""Score fixed-release MemoryAgentBench outputs with the published protocol.

Deterministic metrics follow HUST-AI-HYZ/MemoryAgentBench.  LongMemEval and
InfBench use that repository's official prompts with a caller-selected
OpenAI-compatible judge.  The output is JSONL-resumable and reports the same
macro columns as ICLR 2026 Table 3.

相对于原 scripts/score_memoryagentbench_official.py 的适配：

- datasets 导入改为 ``research.benchmark.datasets``；
- 支持 ``--sampled``（或自动检测）：只对 ``results.<track>.jsonl`` 里出现的
  题目打分，宏平均（AR/TTL/LRU/SF/Overall = 各 source 均值的均值）只在
  抽样覆盖的 scope 上计算，并在 summary 里记录每个 source 的题目/scope
  数，报告可据此标注"抽样"；全量 3671 题时数值与官方口径完全一致；
- judge 调用走 OpenAI 兼容端点，默认 kimi-k3 sz 端点，关闭 thinking
  （``extra_body.chat_template_kwargs.enable_thinking=false``），temperature 0。

用法（repo 根目录）::

    .venv/bin/python -m research.benchmark.score_memoryagentbench_official \
        research/.benchmark_runs/<run> --track pi --sampled
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import string
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

from research.benchmark.datasets import load_memoryagentbench


OFFICIAL_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"
DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
ENTITY2ID_SHA256 = "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
FULL_QUESTIONS = 3671

RESEARCH_ROOT = Path(__file__).resolve().parents[1]


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_answer(text: str) -> str:
    match = re.search(r"(?:Answer:)(.*)(?:\n|$)", text, flags=re.I)
    if match:
        return match.group(1).strip()
    return text.splitlines()[0].strip() if text.splitlines() else text.strip()


def aliases(item: Any) -> list[str]:
    values = list(item.metadata.get("answer_aliases") or [])
    return [str(value) for value in values] or [str(item.answer)]


def deterministic_score(item: Any, prediction: str) -> float:
    source = str(item.metadata.get("official_source") or "")
    golds = aliases(item)
    if source.startswith("icl_") or source == "detective_qa":
        parsed = parse_answer(prediction)
        return float(any(normalize_answer(parsed) == normalize_answer(g) for g in golds))
    parsed = parse_answer(prediction)
    candidates = [prediction, parsed]
    return float(any(
        normalize_answer(g) in normalize_answer(candidate)
        for candidate in candidates for g in golds
    ))


def movie_name(uri: str) -> str:
    value = uri.split("/")[-1].replace("_", " ").replace("-", " ").replace(">", " ")
    value = re.sub(r"\([^()]*\)", "", value)
    return " ".join(value.split())


def movie_key(value: str) -> str:
    value = re.sub(r"\(\d{4}\)", "", value)
    value = re.sub(r"^(?:answer|recommendations?)\s*:\s*", "", value, flags=re.I)
    value = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", value)
    return normalize_answer(value)


class MovieIndex:
    def __init__(self, path: Path):
        if hashlib.sha256(path.read_bytes()).hexdigest() != ENTITY2ID_SHA256:
            raise RuntimeError("MemoryAgentBench entity2id.json fingerprint changed")
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.id_to_name = {int(entity_id): movie_name(uri) for uri, entity_id in raw.items()}
        self.key_to_name: dict[str, str] = {}
        for name in self.id_to_name.values():
            self.key_to_name.setdefault(movie_key(name), name)
        self.keys = list(self.key_to_name)

    def predicted(self, text: str) -> list[str]:
        numbered = re.findall(r"(?:^|\n)\s*\d+[.)]\s*([^\n]+)", text)
        parts = numbered or re.split(r"[,;\n]", text)
        found: list[str] = []
        for part in parts:
            key = movie_key(part.split(" - ", 1)[0].strip(" \t\"'"))
            if not key:
                continue
            name = self.key_to_name.get(key)
            if name is None:
                matches = difflib.get_close_matches(key, self.keys, n=1, cutoff=0.82)
                name = self.key_to_name[matches[0]] if matches else None
            if name and name not in found:
                found.append(name)
            if len(found) >= 10:
                break
        return found

    def recall5(self, item: Any, prediction: str) -> tuple[float, list[str], list[str]]:
        predicted = self.predicted(prediction)
        gold_ids = [int(value) for value in aliases(item)]
        gold = [self.id_to_name[value] for value in gold_ids]
        score = sum(name in predicted[:5] for name in gold) / len(gold)
        return score, predicted, gold


def official_prompts(repo: Path) -> Any:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official scorer commit changed: {commit}")
    source = repo / "llm_based_eval" / "summarization_evaluate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    required = {"fluency_prompt_book", "recall_prompt_book", "precision_prompt_book"}
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in required:
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                values[target.id] = value
    if set(values) != required:
        raise RuntimeError(f"official summarization prompt constants missing: {required - set(values)}")
    return SimpleNamespace(**values)


def longmem_prompt(item: Any, prediction: str) -> str:
    task = str(item.question_type)
    question, answer = item.question, item.answer
    if "_abs" in item.question_id:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information is "
            f"not.\n\nQuestion: {question}\n\nExplanation: {answer}"
            f"\n\nModel Response: {prediction}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        rule = (
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. Equivalent answers or all intermediate steps are correct; a "
            "response containing only a subset of required information is incorrect."
        )
    elif task == "temporal-reasoning":
        rule = (
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. Equivalent answers are correct. Do not penalize off-by-one "
            "errors for numbers of days, weeks, or months."
        )
    elif task == "knowledge-update":
        rule = (
            "Please answer yes if the response contains the correct updated answer. "
            "Previous information may also appear as long as the updated answer is present."
        )
    elif task == "single-session-preference":
        rule = (
            "Please answer yes if the response satisfies the personalized-response rubric. "
            "It need not reflect every rubric point, provided it correctly recalls and uses "
            "the user's personal information."
        )
    else:
        raise ValueError(f"unsupported LongMemEval type: {task}")
    return (
        "I will give you a question, a correct answer or rubric, and a model response. "
        f"{rule}\n\nQuestion: {question}\n\nCorrect Answer: {answer}"
        f"\n\nModel Response: {prediction}\n\nIs the model response correct? "
        "Answer yes or no only."
    )


def parse_json(text: str) -> dict[str, Any]:
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"judge did not return JSON: {text[-500:]}")


class Judge:
    """OpenAI 兼容 judge；temperature 0 且强制关闭 thinking。"""

    def __init__(self, base_url: str, model: str, workers: int, max_tokens: int):
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("KIMI_K3_API_KEY", "not-required"))
        self.model = model
        self.workers = workers
        self.max_tokens = max_tokens

    def call(self, prompt: str, max_tokens: int | None = None) -> str:
        delay = 1.0
        for attempt in range(7):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return str(response.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 6:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 20)
        raise AssertionError("unreachable")


def _call_json(judge: "Judge", request: str, max_tokens: int | None = None) -> str:
    """judge 调用 + 可解析性重试：端点偶发只回 reasoning 纯文本/空 content
    （enable_thinking=false 也拦不住）。两手抓：追加"只回 JSON"指令重试 +
    逐次放大 max_tokens——截空 content 的另一根因是前置 reasoning 变长把
    预算吃光（finish_reason=length），换措辞救不回来，只能加预算。不改评分语义。"""
    last_err: ValueError | None = None
    prompt = request
    budget = max_tokens
    for _ in range(5):
        raw = judge.call(prompt, budget)
        try:
            parse_json(raw)
            return raw
        except ValueError as exc:
            last_err = exc
            prompt = request + (
                "\n\nRespond with ONLY the JSON object."
                " No prose, no markdown, no reasoning."
            )
            if budget is not None:
                budget = min(int(budget) * 2, 16384)
            time.sleep(3)
    assert last_err is not None
    raise last_err


def score_llm(item: Any, prediction: str, judge: Judge, prompts: Any) -> dict[str, Any]:
    source = str(item.metadata.get("official_source") or "")
    if source == "longmemeval_s*":
        raw = judge.call(longmem_prompt(item, prediction))
        return {"score": float("yes" in raw.lower()), "judge_outputs": [raw]}
    if source == "infbench_sum_eng_shots2":
        keypoints = list(item.judge_rubric)
        requests = [
            prompts.fluency_prompt_book.format(text=prediction.strip()),
            prompts.recall_prompt_book.format(
                keypoints="\n".join(f"{i + 1}. {point}" for i, point in enumerate(keypoints)),
                summary=prediction.strip(),
            ),
            prompts.precision_prompt_book.format(
                expert_summary=item.answer, summary=prediction.strip()
            ),
        ]
        # InfBench 三连 judge 沿用官方脚本的 4096 token 上限：本端点即使带
        # enable_thinking=false 也会先输出一段 reasoning，1024 会被截成空
        # content（实测 finish_reason=length），官方取值无此问题。
        raw = [_call_json(judge, request, 4096) for request in requests]
        fluency, recall, precision = [parse_json(value) for value in raw]
        rec = float(recall["recall"]) / len(keypoints) if keypoints else 0.0
        sentences = int(precision["sentence_count"])
        prec = float(precision["precision"]) / sentences if sentences else 0.0
        f1 = float(fluency["fluency"]) * 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        return {
            "score": f1,
            "fluency": float(fluency["fluency"]),
            "recall": rec,
            "precision": prec,
            "judge_outputs": raw,
        }
    raise ValueError(source)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """ICLR Table-3 宏平均；缺席的 source（抽样）记为 None 且不进入均值。"""

    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_source.setdefault(
            row["source"], {"scores": [], "questions": 0, "scopes": set()},
        )
        entry["scores"].append(float(row["score"]))
        entry["questions"] += 1
        if row.get("scope_id"):
            entry["scopes"].add(row["scope_id"])
    source_scores: dict[str, dict[str, Any]] = {
        key: {
            "score": _mean(value["scores"]),
            "questions": value["questions"],
            "scopes": len(value["scopes"]),
        }
        for key, value in sorted(by_source.items())
    }
    score_of = lambda key: source_scores[key]["score"] if key in source_scores else None  # noqa: E731

    def family(prefix_or_keys: Any) -> float | None:
        if isinstance(prefix_or_keys, str):
            values = [score_of(key) for key in source_scores if key.startswith(prefix_or_keys)]
        else:
            values = [score_of(key) for key in prefix_or_keys]
        return _mean([value for value in values if value is not None])

    event, lme = family("eventqa_"), score_of("longmemeval_s*")
    shqa, mhqa = score_of("ruler_qa1_197K"), score_of("ruler_qa2_421K")
    ar = _mean([value for value in (shqa, mhqa, lme, event) if value is not None])
    mcc, recsys = family("icl_"), score_of("recsys_redial_full")
    ttl = _mean([value for value in (mcc, recsys) if value is not None])
    summ, detqa = score_of("infbench_sum_eng_shots2"), score_of("detective_qa")
    lru = _mean([value for value in (summ, detqa) if value is not None])
    fc_sh, fc_mh = family("factconsolidation_sh_"), family("factconsolidation_mh_")
    sf = _mean([value for value in (fc_sh, fc_mh) if value is not None])
    overall = _mean([value for value in (ar, ttl, lru, sf) if value is not None])
    return {
        "source_scores": source_scores,
        "table3": {
            "AR": {"SH-QA": shqa, "MH-QA": mhqa, "LME(S*)": lme, "EventQA": event, "Avg": ar},
            "TTL": {"MCC": mcc, "Recom": recsys, "Avg": ttl},
            "LRU": {"Summ": summ, "DetQA": detqa, "Avg": lru},
            "SF": {"FC-SH": fc_sh, "FC-MH": fc_mh, "Avg": sf},
            "Overall": overall,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--track", required=True)
    parser.add_argument("--official-repo", type=Path,
                        default=RESEARCH_ROOT / ".benchmark_refs" / "MemoryAgentBench")
    parser.add_argument("--entity2id", type=Path,
                        default=RESEARCH_ROOT / ".benchmark_data" / "memoryagentbench" / "entity2id.json")
    parser.add_argument("--base-url", default="http://sz-infer.x2robot.cn/infer/inf-dddgcurffgz3366q/v1")
    parser.add_argument("--judge-model", default="kimi-k3")
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sampled", action="store_true",
                        help="只评 results.<track>.jsonl 中出现的题目（缺省时自动检测）")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = Path(manifest["dataset_path"])
    fixed = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if fixed.get("revision") != DATASET_REVISION or manifest.get("dataset") != "memoryagentbench":
        raise RuntimeError("not the fixed MemoryAgentBench release")
    items = load_memoryagentbench(dataset_manifest)
    by_id = {item.question_id: item for item in items}
    results_path = run_dir / f"results.{args.track}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").split("\n"):  # 只按\n切，记录内可能含\u2028 等
        if line.strip():
            row = json.loads(line)
            latest[str(row["question_id"])] = row
    sampled = args.sampled or set(latest) != set(by_id)
    unknown = set(latest) - set(by_id)
    if unknown:
        raise RuntimeError(f"results contain unknown MemoryAgentBench questions: {sorted(unknown)[:5]}")
    if any(row.get("status") != "completed" for row in latest.values()):
        raise RuntimeError("answer track is not complete")
    if not sampled and len(latest) != FULL_QUESTIONS:
        raise RuntimeError(
            f"answer track is partial ({len(latest)} rows); pass --sampled to score it"
        )
    if not latest:
        raise RuntimeError("no completed answers in results track")

    prompts = official_prompts(args.official_repo.resolve())
    movies = MovieIndex(args.entity2id.resolve())
    judge = Judge(args.base_url, args.judge_model, args.workers, args.judge_max_tokens)
    suffix = ".sampled" if sampled else ""
    output = run_dir / f"memoryagentbench_scores.{args.track}.kimik3-official-v1{suffix}.jsonl"
    completed = {
        str(row["question_id"]): row for row in (
            json.loads(line) for line in output.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ) if row.get("status") == "completed"
    } if output.exists() else {}
    lock = threading.Lock()

    def one(qid: str) -> dict[str, Any]:
        item, answer_row = by_id[qid], latest[qid]
        prediction = str(answer_row.get("hypothesis") or answer_row.get("prediction") or "")
        source = str(item.metadata.get("official_source") or "")
        base = {
            "question_id": qid, "scope_id": item.scope_id,
            "source": source, "status": "completed",
        }
        if source == "recsys_redial_full":
            score, predicted, gold = movies.recall5(item, prediction)
            return {**base, "metric": "Recall@5", "score": score, "predicted_movies": predicted, "gold_movies": gold}
        if source in {"longmemeval_s*", "infbench_sum_eng_shots2"}:
            return {**base, "metric": "LLM-as-judge" if source == "longmemeval_s*" else "HELMET-F1", **score_llm(item, prediction, judge, prompts)}
        metric = "exact_match" if source.startswith("icl_") or source == "detective_qa" else "substring_exact_match"
        return {**base, "metric": metric, "score": deterministic_score(item, prediction)}

    pending = [qid for qid in latest if qid not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(one, qid): qid for qid in pending}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - 单题评分失败不终止
                row = {"question_id": qid, "source": str(by_id[qid].metadata.get("official_source") or ""), "status": "error", "error": repr(exc)}
            with lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("status") == "completed":
                completed[qid] = row

    if len(completed) != len(latest):
        raise RuntimeError(f"scoring incomplete: completed={len(completed)} expected={len(latest)}")
    summary = {
        "dataset": "memoryagentbench",
        "dataset_revision": DATASET_REVISION,
        "track": args.track,
        "sampled": sampled,
        "questions": len(completed),
        "full_questions": FULL_QUESTIONS,
        "scopes": len({by_id[qid].scope_id for qid in completed}),
        "judge_model": args.judge_model,
        "official_scorer_repo": str(args.official_repo.resolve()),
        "official_scorer_commit": OFFICIAL_COMMIT,
        "entity2id_sha256": ENTITY2ID_SHA256,
        **macro_summary(list(completed.values())),
    }
    summary_path = run_dir / f"memoryagentbench_summary.{args.track}.kimik3-official-v1{suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
