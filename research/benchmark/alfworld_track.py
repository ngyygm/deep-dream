"""ALFWorld ReAct 轨道：kimi-k3 等模型在 ALFWorld（TextWorld 后端）里跑
2-shot ReAct 具身任务，按局记录 won/steps/actions/tokens，产出成功率汇总。

重要：本脚本**不能**用仓库 .venv 跑——alfworld/textworld 只装在专用 venv::

    research/.benchmark_runtime/alfworld-venv/bin/python

且必须把 ALFWORLD_DATA 指到旧文法数据目录（0.4.2 自带的新文法数据会打断
"put X in/on Y" 放置动作，详见 probe 结论；旧文法数据已 APFS clone 到::

    research/.benchmark_runtime/alfworld-data   # json_2.1.1 + logic（134 unseen / 140 seen）

用法（仓库根目录）::

    ALFWORLD_DATA=$PWD/research/.benchmark_runtime/alfworld-data \\
    research/.benchmark_runtime/alfworld-venv/bin/python research/benchmark/alfworld_track.py \\
        research/.benchmark_runs/<run_dir> \\
        --config research/service_config.kimi-par.json \\
        --split eval_out_of_distribution --workers 2

全量 unseen 134 局 / seen 140 局就是把上面的 --limit 去掉（数据目录全量注册）。

依赖约束：只用 stdlib + yaml + openai + alfworld，**禁止仓库内模块导入**。
本文件会被没有 alfworld/openai 的仓库 .venv import 做单元测试，因此
openai/alfworld/yaml 一律在函数内延迟导入。

产物（写入 run_dir）：
- results.alfworld.jsonl   每局一条记录，resume 按 gamefile 去重（status=completed 才算 done）
- summary.alfworld.json    总体/分任务类型成功率、平均步数、token 合计
- report.alfworld.md       汇总表
- run_notes.alfworld.json  manifest-lite（split、局数、config sha256、workers 等）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue as queue_mod
import re
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(__file__).resolve().parent / "alfworld_assets"
BASE_CONFIG_PATH = ASSETS_DIR / "react_base_config.yaml"
PROMPTS_PATH = ASSETS_DIR / "alfworld_3prompts.json"

SPLITS = ("eval_out_of_distribution", "eval_in_distribution")

# 游戏目录前缀 -> 2-shot 示例 key 前缀（react_<key>_0/_1，与 probe 验证一致）
TASK_TYPE_PREFIXES: list[tuple[str, str]] = [
    ("pick_and_place", "put"),
    ("pick_clean_then_place", "clean"),
    ("pick_heat_then_place", "heat"),
    ("pick_cool_then_place", "cool"),
    ("look_at_obj", "examine"),
    ("pick_two_obj", "puttwo"),
]

# 'think: ...' 是 ReAct 无操作：env 返回 "Nothing happens."，按惯例喂回 "OK."
NOTHING_FEEDBACK = "Nothing happens."
OBS_CAP_CHARS = 600                      # 单条 observation 进历史/记录的截断长度
TRUNC_KEEP_FIRST = 3                     # 历史超限裁剪时保留的开头步数
TRUNC_KEEP_LAST = 15                     # 历史超限裁剪时保留的结尾步数
TRUNC_MARKER = "[... {n} steps omitted ...]"

# put -> move 安全改写：旧文法的 admissible 命令是 "put {o} in/on {r}"（字面
# in/on），模型偶尔吐 "put X in Y"/"put X into Y"/"put X on Y" 会 Nothing
# happens.；文法里 "move {o} to {r}" 对普通物体恒可用，作为一次性重试兜底。
_PUT_RE = re.compile(r"^put\s+(.+?)\s+(?:in/on|into|in|on)\s+(.+?)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 纯函数区（不依赖 alfworld/openai/yaml，repo .venv 可直接单测）
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """粗估 token 数（英文 ~4 字符/token），仅用于历史裁剪阈值判断。"""
    return max(1, len(text) // 4)


def process_ob(ob: str) -> str:
    """去掉 'You arrive at loc 11. ' 导航前缀（与 ReAct 原实现一致）。"""
    if ob.startswith("You arrive at loc "):
        idx = ob.find(". ")
        if idx != -1:
            return ob[idx + 2:]
    return ob


def strip_banner(obs: str) -> str:
    """首帧去掉 '-= Welcome to TextWorld, ALFRED! =-' 横幅（按空行分段丢弃第一段）。"""
    return "\n".join(obs.split("\n\n")[1:])


def task_type_and_key(gamefile: str) -> tuple[str, str]:
    """从 gamefile 路径取 (任务类型短名, 2-shot 示例 key)。

    路径形如 .../json_2.1.1/valid_unseen/pick_and_place_simple-Remote-.../trial_.../game.tw-pddl，
    逐段找任务目录前缀（不依赖固定层级，split 目录紧跟 json_2.1.1 之后）；
    未识别时回退 ("unknown", "put")。
    """
    for part in Path(gamefile).parts:
        for prefix, key in TASK_TYPE_PREFIXES:
            if part.startswith(prefix):
                return prefix, key
    return "unknown", "put"


def game_short_name(gamefile: str) -> str:
    """局名 = 任务目录/trial 目录（路径最后两段），日志友好。"""
    parts = Path(gamefile).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(gamefile)


def build_system_prompt(prompts: dict[str, str], task_key: str) -> str:
    """按任务类型拼 2-shot 系统提示（示例顺序 _1 在前 _0 在后，与 probe 验证一致）。"""
    return (
        "Interact with a household to solve a task. Here are two examples.\n"
        + prompts[f"react_{task_key}_1"]
        + prompts[f"react_{task_key}_0"]
        + "\nHere is the task.\n"
    )


def parse_action(text: str) -> str:
    """取回复第一个非空行作为动作，剥掉行首 '>' 前缀；空回复返回 ''（调用方回退 look）。"""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        return line.lstrip(">").strip()
    return ""


def put_to_move(action: str) -> str | None:
    """安全改写：'put X in/on|into|in|on Y' -> 'move X to Y'；非 put 命令返回 None。"""
    m = _PUT_RE.match((action or "").strip())
    if not m:
        return None
    return f"move {m.group(1).strip()} to {m.group(2).strip()}"


def truncate_steps(steps: list[str], token_budget: int, system_prompt: str, task_ob: str) -> list[str]:
    """历史超预算时保留首 3 步 + 末 15 步，中段折叠成省略标记。

    判断口径：system + 任务首帧 + 历史拼起来的粗估 token 总量；
    steps 的每个元素已经是 '> action\\nobs\\n' 块。
    """
    joined = system_prompt + "\n" + task_ob + "\n" + "".join(steps)
    if estimate_tokens(joined) <= token_budget:
        return steps
    if len(steps) <= TRUNC_KEEP_FIRST + TRUNC_KEEP_LAST:
        return steps  # 步数少但超限（单步超长场景）：原样返回，靠 OBS_CAP 兜底
    omitted = len(steps) - TRUNC_KEEP_FIRST - TRUNC_KEEP_LAST
    return steps[:TRUNC_KEEP_FIRST] + [TRUNC_MARKER.format(n=omitted) + "\n"] + steps[-TRUNC_KEEP_LAST:]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总记录：总体/分任务类型 SR、平均步数、token 合计（只统计 completed）。"""
    completed = [r for r in records if r.get("status") == "completed"]
    won_recs = [r for r in completed if r.get("won")]

    def _bucket(recs: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(recs)
        w = len([r for r in recs if r.get("won")])
        steps = [int(r.get("steps", 0)) for r in recs]
        return {
            "games": n,
            "won": w,
            "success_rate": round(w / n, 4) if n else None,
            "avg_steps": round(sum(steps) / n, 2) if n else None,
        }

    per_type: dict[str, list[dict[str, Any]]] = {}
    for r in completed:
        per_type.setdefault(r.get("task_type", "unknown"), []).append(r)
    summary: dict[str, Any] = {
        "overall": _bucket(completed),
        "per_task_type": {k: _bucket(v) for k, v in sorted(per_type.items())},
        "avg_steps_won": (
            round(sum(int(r["steps"]) for r in won_recs) / len(won_recs), 2) if won_recs else None
        ),
        "tokens": {
            "prompt": sum(int(r.get("prompt_tokens", 0)) for r in completed),
            "completion": sum(int(r.get("completion_tokens", 0)) for r in completed),
        },
        "games_completed": len(completed),
        "games_error": len([r for r in records if r.get("status") != "completed"]),
    }
    summary["tokens"]["total"] = summary["tokens"]["prompt"] + summary["tokens"]["completion"]
    return summary


def render_report(summary: dict[str, Any], *, split: str, model: str, run_dir: Path) -> str:
    """把 summary 渲染成 markdown 汇总表（report.alfworld.md）。"""
    lines = [
        "# ALFWorld ReAct 轨道报告",
        "",
        f"- split: `{split}`",
        f"- model: `{model}`",
        f"- run_dir: `{run_dir}`",
        f"- games completed: {summary['games_completed']}（error {summary['games_error']}）",
        "",
        "| 任务类型 | 局数 | 胜局 | SR | 平均步数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task_type, bucket in summary["per_task_type"].items():
        sr = "-" if bucket["success_rate"] is None else f"{bucket['success_rate'] * 100:.1f}%"
        avg = "-" if bucket["avg_steps"] is None else f"{bucket['avg_steps']:.1f}"
        lines.append(f"| {task_type} | {bucket['games']} | {bucket['won']} | {sr} | {avg} |")
    overall = summary["overall"]
    lines.append(
        f"| **overall** | {overall['games']} | {overall['won']} "
        f"| **{overall['success_rate'] * 100:.1f}%** | {overall['avg_steps']:.1f} |"
    )
    tok = summary["tokens"]
    avg_won = "-" if summary["avg_steps_won"] is None else f"{summary['avg_steps_won']:.1f}"
    lines += [
        "",
        f"- 平均步数（胜局）: {avg_won}",
        f"- tokens: prompt {tok['prompt']} / completion {tok['completion']} / total {tok['total']}",
        "",
    ]
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# LLM 与环境（延迟导入 openai/alfworld/yaml）
# ---------------------------------------------------------------------------

def load_llm_config(config_path: Path) -> dict[str, Any]:
    """从 service config json 读 llm 设置（api_key 只进内存，绝不打印）。"""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    llm = cfg.get("llm") or {}
    return {
        "base_url": llm["base_url"],
        "api_key": llm.get("api_key", "EMPTY"),
        "model": llm["model"],
        "temperature": float(llm.get("temperature", 0.0)),
        "max_tokens": int(llm.get("max_tokens", 16384)),
        "context_window_tokens": int(llm.get("context_window_tokens", 32000)),
        "timeout_seconds": float(llm.get("timeout_seconds", 300)),
        "extra_body": llm.get("extra_body") or {"chat_template_kwargs": {"enable_thinking": False}},
    }


def _chat(client: Any, llm: dict[str, Any], system: str, user: str) -> tuple[str, dict[str, int]]:
    """一次 chat.completions 调用（温度 0、关思考），带 3 次退避重试；返回 (文本, usage)。"""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=llm["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=llm["temperature"],
                max_tokens=llm["max_tokens"],
                extra_body=llm["extra_body"],
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            return text, {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception as exc:  # noqa: BLE001 - 网络/限流抖动，退避后重试
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_err}")


def _step_with_adapter(env_game: Any, action: str) -> dict[str, Any]:
    """执行一个动作；think: 观测固定为 'OK.'，put 类 Nothing happens. 时用 move 形式重试一次。"""
    obs, _scores, dones, infos = env_game.step([action])
    ob_text = process_ob(obs[0])
    executed, adapted = action, False
    if ob_text == NOTHING_FEEDBACK and not action.startswith("think:"):
        alt = put_to_move(action)
        if alt:
            obs, _scores, dones, infos = env_game.step([alt])
            ob_text = process_ob(obs[0])
            executed, adapted = f"{action} !!-> {alt}", True
    return {
        "ob_text": ob_text[:OBS_CAP_CHARS],
        "executed": executed,
        "is_think": action.startswith("think:"),
        "adapted": adapted,
        "dones": dones,
        "infos": infos,
    }


def play_game(env_game: Any, client: Any, llm: dict[str, Any], prompts: dict[str, str],
              gamefile: str, first_obs: str, split: str, max_steps: int,
              token_budget: int, worker_id: int) -> dict[str, Any]:
    """单局 ReAct 循环（reset 已由调用方完成，首帧 obs 传入）。

    - 首动作 "look"（canonical ReAct warm-up，与 probe 验证一致）
    - 每步：LLM 回复取第一个非空行（剥 '> '），空回复回退 "look"
    - think: 无操作 -> 观测记 "OK."
    - 历史粗估超 token_budget 时折叠中段
    """
    task_type, task_key = task_type_and_key(gamefile)
    system_prompt = build_system_prompt(prompts, task_key)
    record: dict[str, Any] = {
        "track": "alfworld",
        "split": split,
        "status": "error",
        "gamefile": str(gamefile),
        "game": game_short_name(str(gamefile)),
        "task_type": task_type,
        "task_key": task_key,
        "model": llm["model"],
        "worker": worker_id,
        "won": 0,
        "steps": 0,
        "env_steps": 0,
        "adapted_puts": 0,
        "actions": [],
        "observations": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
    }
    task_ob = strip_banner(first_obs)
    steps_history: list[str] = []
    action = "look"
    won = 0
    started = time.monotonic()
    try:
        for step_idx in range(1, max_steps + 1):
            outcome = _step_with_adapter(env_game, action)
            ob_text = "OK." if outcome["is_think"] else outcome["ob_text"]
            record["env_steps"] += 2 if outcome["adapted"] else 1
            record["adapted_puts"] += int(outcome["adapted"])
            record["actions"].append(outcome["executed"])
            record["observations"].append(ob_text)
            steps_history.append(f"> {outcome['executed']}\n{ob_text}\n")
            won = int(bool(outcome["infos"]["won"][0]))
            record["steps"] = step_idx
            if outcome["dones"][0]:
                break
            blocks = truncate_steps(steps_history, token_budget, system_prompt, task_ob)
            user = task_ob + "\n" + "".join(blocks) + ">"
            text, usage = _chat(client, llm, system_prompt, user)
            record["prompt_tokens"] += usage["prompt_tokens"]
            record["completion_tokens"] += usage["completion_tokens"]
            record["llm_calls"] += 1
            action = parse_action(text) or "look"
        record["won"] = won
        record["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - 单局失败记 error，不拖垮 worker
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    record["latency_seconds"] = round(time.monotonic() - started, 2)
    return record


def _worker_main(worker_id: int, games: list[str], done: list[str], llm: dict[str, Any],
                 split: str, max_steps: int, token_budget: int, out_queue: Any) -> None:
    """spawn worker：构建 AlfredTWEnv 一次，只注册自己 game_files[i::W] 切片，逐局顺序打。

    reset() 后从 infos['extra.gamefile'] 读实际局名（环境内部顺序可能与传入顺序
    不同），以实际 gamefile 为准记录/去重；done 列表里的局直接跳过。
    """
    try:
        import yaml

        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        out_queue.put({"kind": "fatal", "worker": worker_id,
                       "error": f"import failed (用错解释器？需 alfworld venv): {exc}"[:2000]})
        out_queue.put({"kind": "worker_done", "worker": worker_id})
        return

    done_set = set(done)
    try:
        with open(BASE_CONFIG_PATH, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        env = AlfredTWEnv(config, train_eval=split)
        env.game_files = list(games)      # 关键：先切片再 init_env
        env.num_games = len(games)
        env_game = env.init_env(batch_size=1)
        prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        client = OpenAI(base_url=llm["base_url"], api_key=llm["api_key"],
                        timeout=llm["timeout_seconds"], max_retries=2)
        for _ in range(len(games)):
            # ALFWorld 0.4.2: reset() -> (obs, infos) 二元组，step([a]) -> 四元组
            obs, infos = env_game.reset()
            gamefile = str(infos["extra.gamefile"][0])
            if gamefile in done_set:
                out_queue.put({"kind": "skip", "worker": worker_id, "gamefile": gamefile})
                continue
            record = play_game(env_game, client, llm, prompts, gamefile, obs[0],
                               split, max_steps, token_budget, worker_id)
            out_queue.put({"kind": "result", "worker": worker_id, "record": record})
    except Exception as exc:  # noqa: BLE001
        out_queue.put({
            "kind": "fatal", "worker": worker_id,
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:],
        })
    finally:
        out_queue.put({"kind": "worker_done", "worker": worker_id})


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _collect_game_files(split: str) -> list[str]:
    """主进程构建一次 AlfredTWEnv 拿全量 game 列表（排序保证 resume/limit 可复现）。"""
    import yaml

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    with open(BASE_CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    env = AlfredTWEnv(config, train_eval=split)
    return sorted(str(g) for g in env.game_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ALFWorld ReAct track (MUST run under research/.benchmark_runtime/alfworld-venv)")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "research/service_config.kimi-par.json")
    parser.add_argument("--split", choices=SPLITS, default="eval_out_of_distribution")
    parser.add_argument("--limit", type=int, default=None, help="只打排序后的前 N 局（默认全量）")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--trunc-tokens", type=int, default=20000,
                        help="历史裁剪 token 阈值（自动再夹到 context_window-max_tokens-2000）")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="默认开：results 里 completed 的局跳过")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)

    data_dir = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_dir or not (Path(data_dir) / "json_2.1.1").exists():
        print(
            "ERROR: ALFWORLD_DATA 未设置或不含 json_2.1.1（需要旧文法数据目录）。在仓库根目录执行：\n"
            "  export ALFWORLD_DATA=$PWD/research/.benchmark_runtime/alfworld-data",
            file=sys.stderr,
        )
        return 2

    os.environ.setdefault("TQDM_DISABLE", "1")  # 压掉 AlfredTWEnv 收集游戏的 tqdm 噪声

    llm = load_llm_config(args.config.resolve())
    # 历史裁剪阈值：规格默认 20k，但必须给生成留出窗口 -> 夹到 context_window - max_tokens - 2000
    token_budget = min(args.trunc_tokens,
                       max(4000, llm["context_window_tokens"] - llm["max_tokens"] - 2000))

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.alfworld.jsonl"

    all_games = _collect_game_files(args.split)
    if args.limit:
        all_games = all_games[: args.limit]

    done: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").split("\n"):  # 只按\n切，记录内可能含\u2028 等
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "completed" and rec.get("split") == args.split:
                done.add(rec["gamefile"])

    pending = [g for g in all_games if g not in done]
    workers = max(1, min(args.workers, len(pending) or 1))
    print(f"alfworld track: split={args.split} games={len(all_games)} "
          f"(done={len(done)}, pending={len(pending)}) workers={workers} model={llm['model']}")
    print(f"  results -> {results_path}")
    if not pending:
        _write_outputs(run_dir, results_path, args.split, llm, args, len(all_games), 0.0)
        print("All games already done.")
        return 0

    # manifest-lite 运行笔记（split、局数、config hash）
    notes = {
        "track": "alfworld",
        "split": args.split,
        "game_count": len(all_games),
        "pending": len(pending),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256_file(args.config.resolve()),
        "model": llm["model"],
        "base_url": llm["base_url"],
        "max_steps": args.max_steps,
        "token_budget": token_budget,
        "workers": workers,
        "alfworld_data": data_dir,
        "interpreter": sys.executable,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "run_notes.alfworld.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    # spawn 多进程：每个 worker 自建 AlfredTWEnv，吃 game_files[i::W] 切片
    ctx = get_context("spawn")
    out_queue = ctx.Queue()
    slices = {w: pending[w::workers] for w in range(workers)}
    procs = []
    for w in range(workers):
        if not slices[w]:
            continue
        p = ctx.Process(target=_worker_main, args=(
            w, slices[w], sorted(done), llm, args.split, args.max_steps, token_budget, out_queue))
        p.start()
        procs.append(p)

    counters = {"finished": 0, "errors": 0, "fatal": 0}
    finished_workers = 0
    started = time.monotonic()
    while finished_workers < len(procs):
        if not any(p.is_alive() for p in procs):
            # 全部进程已退出（含硬崩溃）：抽干队列后收尾
            while True:
                try:
                    msg = out_queue.get_nowait()
                except queue_mod.Empty:
                    break
                finished_workers, counters = _handle_message(
                    msg, results_path, len(pending), finished_workers, counters)
            break
        try:
            msg = out_queue.get(timeout=5.0)
        except queue_mod.Empty:
            continue
        finished_workers, counters = _handle_message(
            msg, results_path, len(pending), finished_workers, counters)
    for p in procs:
        p.join(timeout=120)

    wall = time.monotonic() - started
    _write_outputs(run_dir, results_path, args.split, llm, args, len(all_games), wall)
    print(f"Done: {counters['finished'] - counters['errors']} completed, "
          f"{counters['errors']} errors, wall={wall:.0f}s -> {results_path}")
    return 0 if counters["errors"] == 0 and counters["fatal"] == 0 else 1


def _handle_message(msg: dict[str, Any], results_path: Path, total: int,
                    finished_workers: int, counters: dict[str, int]) -> tuple[int, dict[str, int]]:
    """父进程处理 worker 消息：落盘 + 进度行。"""
    kind = msg.get("kind")
    if kind == "result":
        rec = msg["record"]
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counters["finished"] += 1
        if rec.get("status") != "completed":
            counters["errors"] += 1
        won = "won" if rec.get("won") else "LOST"
        print(f"[{counters['finished']}/{total}] w{msg.get('worker')} {rec.get('game')} "
              f"{won} steps={rec.get('steps')} tok={rec.get('prompt_tokens', 0)}/"
              f"{rec.get('completion_tokens', 0)} ({rec.get('latency_seconds', 0)}s)"
              + (f" ERR={rec.get('error', '')[:120]}" if rec.get("error") else ""),
              flush=True)
    elif kind == "skip":
        counters["finished"] += 1
        print(f"[{counters['finished']}/{total}] w{msg.get('worker')} skip (done) "
              f"{game_short_name(msg.get('gamefile', ''))}", flush=True)
    elif kind == "fatal":
        counters["fatal"] += 1
        print(f"[worker {msg.get('worker')} FATAL] {msg.get('error', '')[:500]}", file=sys.stderr,
              flush=True)
    elif kind == "worker_done":
        finished_workers += 1
    return finished_workers, counters


def _write_outputs(run_dir: Path, results_path: Path, split: str, llm: dict[str, Any],
                   args: argparse.Namespace, game_count: int, wall_seconds: float) -> None:
    """收尾：summary.alfworld.json + report.alfworld.md（按本次 split 过滤全部记录）。"""
    records: list[dict[str, Any]] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").split("\n"):  # 只按\n切，记录内可能含\u2028 等
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("split") == split:
                records.append(rec)
    summary = summarize_records(records)
    summary.update({
        "track": "alfworld",
        "split": split,
        "model": llm["model"],
        "max_steps": args.max_steps,
        "workers": args.workers,
        "games_registered": game_count,
        "wall_seconds": round(wall_seconds, 1),
    })
    (run_dir / "summary.alfworld.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.alfworld.md").write_text(
        render_report(summary, split=split, model=llm["model"], run_dir=run_dir), encoding="utf-8")
    overall = summary["overall"]
    print(f"summary: SR={overall['success_rate']} ({overall['won']}/{overall['games']}) "
          f"avg_steps={overall['avg_steps']} -> summary.alfworld.json / report.alfworld.md")


if __name__ == "__main__":
    sys.exit(main())
