=== LME full v1/v2 运行时引擎改动证据 ===

## 背景
longmemeval-kimik3-full-v1 (created 2026-08-24T03:18Z) 与 v2 (created 2026-08-27T05:49Z) 的 run_manifest.json 均记录 git_commit=73d9c5e6996aad48c981068056b34d15388f99f9。
v2 运行时工作区含未提交的 v2 引擎改动（manifest remember 配置可证：window_batch_alignment=true / cluster_convergence=true，v1 为 None），该改动三日后才提交。

## 提交链（73d9c5e → 8fa9651，中间仅隔 cee9faf 纯 research/ 迁移）
8fa9651 2026-08-29 19:24:41 +0800 feat(align-v2): 簇收敛引擎转默认 + 四项吞吐修复（V2 全量对比胜出）
cee9faf 2026-08-29 19:23:52 +0800 chore(research): 评测与 agent 运行时迁入 research/（预暂存重命名落地）

## 本目录文件
- engine_patch_73d9c5e_to_8fa9651_core.diff: git diff 73d9c5e..8fa9651 -- core/（43 文件 +3304/−9310）
- service_configs_redacted/: 主仓根目录全部 service_config*.json（api_key 等已脱敏）

## 口径与注意
1. 该 diff 是事后重建：8fa9651 除 v2 引擎外还含四项吞吐修复（json_object_mode/align_llm_cap 等），v2 run 后半程可能已生效、前半程未必——引擎行为以 manifest 内嵌 config 为权威（base_url/model/全 pipeline 参数逐字段落库，且带 runtime_policy 与 SKILL.md 的 sha256 锚点）。
2. A/B 对齐实验（2026-08-25，exp-align-a/b1/b2）的实际运行代码在 /home/linkco/deep-dream/dd-exp/（沙箱非 git 仓，core/ 即当时代码原件），生效配置为 dd-exp/research/service_config.exp-{A,B1,B2}.json。
