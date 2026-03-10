# 示例：记录一段工作

按时间线记录任务过程、搜索、决策、产出。适合会话结束或心跳时批量写入。

## 请求体（POST /api/remember）

```json
{
  "text": "[2026-03-10 14:00-16:00] Kael — 工作记录\n\n14:00 用户要求我调研 RAG 框架。\n14:15 我搜索了 LangChain、LlamaIndex、Haystack 三个框架，比较了文档质量和社区活跃度。\n14:40 我认为 LlamaIndex 更适合当前项目，因为索引抽象更灵活，对本地模型支持更好。\n15:00 开始撰写调研报告。\n15:30 用户决定采用 LlamaIndex。\n15:45 我搭建了 demo 项目，完成基本文档导入和查询流程。\n16:00 demo 跑通，用户满意。\n\n我的思考：LlamaIndex 的 node parser 设计优雅，后续可深入研究自定义 retriever。",
  "source_name": "工作记录",
  "event_time": "2026-03-10T14:00:00"
}
```

要点：时间前缀 + Agent 名 + 活动类型；正文按时间点叙事；结尾可加「我的思考」。
