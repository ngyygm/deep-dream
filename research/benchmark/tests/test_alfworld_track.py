"""alfworld_track 纯函数单测（repo .venv 可跑——模块不允许顶层 import alfworld/openai）。"""
import json
from pathlib import Path

from research.benchmark.alfworld_track import (
    OBS_CAP_CHARS,
    ASSETS_DIR,
    build_system_prompt,
    estimate_tokens,
    game_short_name,
    parse_action,
    process_ob,
    put_to_move,
    render_report,
    strip_banner,
    summarize_records,
    task_type_and_key,
    truncate_steps,
)

GAMEFILE = (
    "/tmp/alfworld-data/json_2.1.1/valid_unseen/"
    "pick_and_place_simple-RemoteControl-None-ArmChair-230/trial_T20190909_021000_274398/game.tw-pddl"
)


def test_module_imports_without_alfworld():
    # 顶层 import 本身就是证明：repo .venv 没装 alfworld/openai
    import research.benchmark.alfworld_track as mod
    assert mod.SPLITS == ("eval_out_of_distribution", "eval_in_distribution")


def test_process_ob_strips_arrival_prefix():
    assert process_ob("You arrive at loc 11. On the countertop 1, you see a mug 2.") == (
        "On the countertop 1, you see a mug 2."
    )
    assert process_ob("On the fridge 1, you see nothing.") == "On the fridge 1, you see nothing."


def test_strip_banner_drops_first_segment():
    raw = "-= Welcome to TextWorld, ALFRED! =-\n\nYour task is to: put X.\n\nYou are in a room."
    # 分段重join后段间只剩单个换行（与 probe 验证的剥横幅实现一致）
    assert strip_banner(raw) == "Your task is to: put X.\nYou are in a room."


def test_task_type_and_key_all_six_types():
    cases = {
        "pick_and_place_simple-Remote-None-Chair-1": ("pick_and_place", "put"),
        "pick_clean_then_place_in_recep-Tomato-None-Sink-1": ("pick_clean_then_place", "clean"),
        "pick_heat_then_place_in_recep-Potato-None-Microwave-1": ("pick_heat_then_place", "heat"),
        "pick_cool_then_place_in_recep-Tomato-None-Fridge-1": ("pick_cool_then_place", "cool"),
        "look_at_obj_in_light-Pencil-None-FloorLamp-1": ("look_at_obj", "examine"),
        "pick_two_obj_and_place-Pencil-None-Sofa-1": ("pick_two_obj", "puttwo"),
    }
    for dirname, expected in cases.items():
        gf = f"/x/json_2.1.1/valid_seen/{dirname}/trial_T1/game.tw-pddl"
        assert task_type_and_key(gf) == expected
    assert task_type_and_key("/x/other/whatever/game.tw-pddl") == ("unknown", "put")


def test_game_short_name():
    assert game_short_name(GAMEFILE) == (
        "pick_and_place_simple-RemoteControl-None-ArmChair-230/trial_T20190909_021000_274398"
    )


def test_build_system_prompt_uses_vendored_assets_for_all_task_keys():
    prompts = json.loads((ASSETS_DIR / "alfworld_3prompts.json").read_text(encoding="utf-8"))
    for _, key in [("a", "put"), ("b", "clean"), ("c", "heat"), ("d", "cool"),
                   ("e", "examine"), ("f", "puttwo")]:
        sys_prompt = build_system_prompt(prompts, key)
        # 示例顺序 _1 在前 _0 在后，与 probe 验证一致
        assert prompts[f"react_{key}_1"] in sys_prompt
        assert prompts[f"react_{key}_0"] in sys_prompt
        assert sys_prompt.startswith("Interact with a household to solve a task.")
        assert sys_prompt.index(prompts[f"react_{key}_1"]) < sys_prompt.index(prompts[f"react_{key}_0"])
        assert sys_prompt.endswith("Here is the task.\n")


def test_parse_action_first_nonempty_line_and_gt_strip():
    assert parse_action("> go to fridge 1") == "go to fridge 1"
    assert parse_action("\n\n   > take mug 1 from countertop 1  \nthink: done") == "take mug 1 from countertop 1"
    assert parse_action("put apple 3 in/on sidetable 1") == "put apple 3 in/on sidetable 1"
    assert parse_action("   \n\n") == ""


def test_put_to_move_adapter():
    assert put_to_move("put apple 3 in/on sidetable 1") == "move apple 3 to sidetable 1"
    assert put_to_move("put mug 1 in box 1") == "move mug 1 to box 1"
    assert put_to_move("put potato 1 into microwave 1") == "move potato 1 to microwave 1"
    assert put_to_move("Put Tomato 2 On fridge 1") == "move Tomato 2 to fridge 1"
    assert put_to_move("go to fridge 1") is None
    assert put_to_move("take apple 1 from fridge 1") is None
    assert put_to_move("think: I need to put it somewhere") is None


def test_truncate_steps_under_budget_unchanged():
    steps = ["> look\nYou see a fridge.\n"] * 10
    assert truncate_steps(steps, token_budget=10_000, system_prompt="sys", task_ob="ob") == steps


def test_truncate_steps_over_budget_keeps_head_and_tail():
    steps = [f"> act {i}\n" + "x" * 40 + "\n" for i in range(25)]
    out = truncate_steps(steps, token_budget=5, system_prompt="sys", task_ob="ob")
    assert len(out) == 3 + 1 + 15
    assert out[:3] == steps[:3]
    assert out[-15:] == steps[-15:]
    assert "[... 7 steps omitted ...]" in out[3]


def test_truncate_steps_short_list_over_budget_untouched():
    steps = ["> act\n" + "y" * 200 + "\n" for _ in range(5)]
    assert truncate_steps(steps, token_budget=1, system_prompt="s", task_ob="o") == steps


def test_estimate_tokens_heuristic():
    assert estimate_tokens("abcd" * 100) == 100
    assert estimate_tokens("") == 1


def test_obs_cap_constant():
    assert OBS_CAP_CHARS == 600


def _rec(**kw):
    base = {
        "track": "alfworld", "status": "completed", "won": 0, "steps": 10,
        "task_type": "pick_and_place", "prompt_tokens": 0, "completion_tokens": 0,
    }
    base.update(kw)
    return base


def test_summarize_records_overall_and_per_type():
    records = [
        _rec(won=1, steps=8, prompt_tokens=100, completion_tokens=20),
        _rec(won=0, steps=50, prompt_tokens=50, completion_tokens=10),
        _rec(won=1, steps=12, task_type="pick_heat_then_place", prompt_tokens=70, completion_tokens=30),
        _rec(status="error", error="boom"),
    ]
    s = summarize_records(records)
    assert s["games_completed"] == 3
    assert s["games_error"] == 1
    assert s["overall"]["won"] == 2
    assert s["overall"]["games"] == 3
    assert abs(s["overall"]["success_rate"] - 2 / 3) < 1e-4
    assert s["overall"]["avg_steps"] == round((8 + 50 + 12) / 3, 2)
    assert s["per_task_type"]["pick_and_place"]["success_rate"] == 0.5
    assert s["per_task_type"]["pick_heat_then_place"]["success_rate"] == 1.0
    assert s["avg_steps_won"] == 10.0
    assert s["tokens"] == {"prompt": 220, "completion": 60, "total": 280}


def test_render_report_markdown_table():
    records = [
        _rec(won=1, steps=8),
        _rec(won=0, steps=50),
        _rec(won=1, steps=12, task_type="pick_heat_then_place"),
    ]
    s = summarize_records(records)
    s.update({"games_registered": 3, "wall_seconds": 12.0, "max_steps": 50, "workers": 2})
    md = render_report(s, split="eval_out_of_distribution", model="kimi-k3", run_dir=Path("/tmp/r"))
    assert "| pick_and_place | 2 | 1 | 50.0% | 29.0 |" in md
    assert "| pick_heat_then_place | 1 | 1 | 100.0% | 12.0 |" in md
    assert "**66.7%**" in md
    assert "eval_out_of_distribution" in md and "kimi-k3" in md
