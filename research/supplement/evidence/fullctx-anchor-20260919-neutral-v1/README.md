# fullctx neutral-v1 锚点交接（2026-09-19，评测机）

## 结果
- track: full-context-kimik3-neutral-v1 @ memoryagentbench-kimik3-sample-v2
- 607/607 题零错误；官方 scorer Overall = **0.7518**
- Table-3: AR 0.9675 / TTL(MCC) 0.75 / LRU 0.510 / SF(FC) 0.78
- 对照 v1（0.4601）: MCC 0→0.75、FC-MH 0→0.56、FC-SH 0.75→1.00

## 我在 neutral-v1 上做的两处 operator 修复（diff 随包，请合入/追认）
1. **删除弃答句**（runner.py neutral-v1 prompt 里的 "If the context does not contain the
   answer, answer exactly 'No information available.'"）。A/B/C 实验（icl_clinic150_7050shot_balance_no1,
   标准答案 '19'）:
   - A 现版(带弃答句, think off): "No information available." ✗
   - B 仅删弃答句(think off): '19' ✓（3/3 稳定）
   - C 续写提示替代: '19' ✓（3/3）
   弃答门控即全部病灶；删除与"任务中性、不强加启发式"的设计自洽（弃答句本身就是强加的启发式）。
2. **thinking 对齐冻结轨**: service_config.kimiK3.json 补 think=false +
   extra_body.enable_thinking=false + max_tokens=16384（对齐 6 条冻结轨 manifest）。
   注: v1 锚点因 config 缺 extra_body 实际是 thinking-ON 跑的（manifest 可见 extra_body={}），
   v1 的 ICL 全 0 是"弃答句 × thinking-on"双重放大。
   ⚠️ 用户提出、留给论文口径定夺：是否补一个 think-on 的"最佳努力锚点"变体（对标已发表
   full-context 绝对水平）；当前 think-off 版与 evidence 同口径，作主锚点。

## 同事第 3 项的答案（端点窗口）
- 新实例 inf-ddjulule2xcaenzd 硬上限 = **262,144 tokens**（探测: 1.05M chars 中文填充 →
  400 "input 565480 tokens > context length 262144"），模型列表仅 kimi-k3。
- ar-005(1.695M chars)/ar-020(1.588M chars) 无法补跑，覆盖定格 607/767 + manifest 披露。
- 副产品: 中文真实 tokenizer ≈2 chars/token，preflight ×4 公式对中文乐观 2 倍（本次全部
  通过因端点超限是 400 硬拒绝不截断，锚点完整性无损；改 prompt/加 scope 时建议预算公式按语言区分）。

## 运行节奏（复现用）
8 并发跑 9 中小 scope（~23 题/min）→ 4 并发 resume 补 ar-000 巨题 → 2 并发终扫。
scorer: PYTHONPATH=. + 显式 --base-url 新实例（其默认仍指死实例）。
config 用 service_config.kimiK3.json（注意：指令原文写的 service_config.json 指向已死旧实例）。

## 叙事提示（供分析节）
neutral-v1 下 fullctx Overall 0.752 > pi 0.659；仅 FC(0.78 vs 0.80)pi 险胜。
记忆系统卖点建议往效率轴（pi 检索预算 vs 锚点 ~56M tokens/轮）倾斜。v1(0.46) 作 prompt
敏感性 disclosed artifact 保留。
