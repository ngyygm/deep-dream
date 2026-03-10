# 示例：记录想法与决策（当日反思）

记录当天的任务、决策理由、思考与明日计划。适合每日结束或心跳时写入。

## 请求体（POST /api/remember）

```json
{
  "text": "[2026-03-10 22:00] Kael — 当日反思\n\n今天处理了三个任务：RAG 调研、TMG 接口优化、文档校对。\n\n关于 RAG：LlamaIndex 长文档分块策略比 LangChain 合理，但 LangChain 的 Agent 生态更成熟。后续可能混合使用。\n\n关于 TMG：event_time 解决了根本问题——之前所有记忆时间都是「处理时间」而非「发生时间」，时间回溯会产生偏差。\n\n明天计划：LlamaIndex demo 加入自定义 retriever；测试 TMG find 的时间过滤准确性。",
  "source_name": "每日反思",
  "event_time": "2026-03-10T22:00:00"
}
```

要点：任务列表 + 每项的决策/理由 + 明日计划；保持自然语言，便于日后 find 召回。
