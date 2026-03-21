# 示例：记录想法与决策（当日反思）

记录当天的任务、决策理由、思考与明日计划。适合每日结束或心跳时写入。

## 查询参数（GET /api/remember）

```bash
curl -s -G "http://127.0.0.1:16200/api/remember" \
  --data-urlencode "text=[2026-03-10 22:00] Kael — 当日反思

今天处理了三个任务：RAG 调研、TMG 接口优化、文档校对。
……" \
  --data-urlencode "source_name=每日反思" \
  --data-urlencode "event_time=2026-03-10T22:00:00"
```

要点：任务列表 + 每项的决策/理由 + 明日计划；保持自然语言，便于日后 find 召回。
