# fullctx-anchor 轨道运行交接（2026-09-19，评测机）

## 结果速览
- track: full-context-kimik3-v1 @ memoryagentbench-kimik3-sample-v2
- 覆盖: 607/767 题（8/10 冻结 scope）；mab-ar-005、mab-ar-020 超 1.04M chars 预算被预检排除（160 题，manifest excluded_scopes 有记录，可披露）
- 官方 scorer Overall = 0.4601（--sampled；summary json 随包）
- config: context_window_tokens=262144, timeout_seconds=1800, kimi-k3

## 运行环境事实（同事须知）
1. **端点已换实例**: 旧 inf-dddgcurffgz3366q 已下线（404 Route Not Found），现用
   inf-ddjulule2xcaenzd（curl 直连验证可用，247k-token 巨题 23s 返回）。
   scorer 的 --base-url 默认值仍指旧实例，需显式传新地址（judge 调用才不走死路由）。
2. **并发经验**: 16 并发全 504 Gateway Time-out（且提交序里 ar-000 巨题排最前会占死全部
   worker）；8 并发对中小题稳定（~18 题/min）；ar-000 的 197K ruler 巨题需 ≤4 并发。
   实跑节奏: 8 并发跑 9 中小 scope → 4 并发 resume 补 ar-000 → 2 并发终扫，607/607 零错误。
3. scorer 需 `PYTHONPATH=.` 且用仓库 .venv（pyarrow 等依赖）。

## operator 本地补丁（随包 diff，请合入分支）
- full_context.py: dataset 目录解析改为从 manifest.dataset_path 向上回溯到注册表 filename
  命中处——原实现固定取 parent，MAB 嵌套布局下双重拼接直接 FileNotFoundError
- cli.py + full_context.py: 新增 --scope-id 多值过滤，用于锚点对齐冻结采样 scope
  （原实现默认跑全集 3671 题 ≈ 222M tokens；采样口径 767 题 ≈ 56M tokens）

## 文件清单
- ../research/.benchmark_runs/memoryagentbench-kimik3-sample-v2/ 下 4 个工件:
  results.full-context-kimik3-v1.jsonl（687 行含重试史，latest 607）
  memoryagentbench_scores.full-context-kimik3-v1.kimik3-official-v1.sampled.jsonl（607，0 错）
  memoryagentbench_summary.full-context-kimik3-v1.kimik3-official-v1.sampled.json
  run_manifest.json（tracks 含 full-context-kimik3-v1，track_variants 带完整 full_context 溯源）
- service_config.kimiK3.redacted.json（脱敏；timeout_seconds=1800 已入）
- operator_patches_fullctx_scope_and_datadir.diff
