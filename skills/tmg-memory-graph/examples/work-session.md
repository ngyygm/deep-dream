# 示例：记录一段工作

按时间线记录任务过程、搜索、决策、产出。适合会话结束或心跳时批量写入。

## 查询参数（GET /api/remember）

```bash
curl -s -G "http://127.0.0.1:16200/api/remember" \
  --data-urlencode "text=[2026-03-10 14:00-16:00] Kael — 工作记录

14:00 用户要求我调研 RAG 框架。
……
我的思考：……" \
  --data-urlencode "source_name=工作记录" \
  --data-urlencode "event_time=2026-03-10T14:00:00"
```

要点：时间前缀 + Agent 名 + 活动类型；正文按时间点叙事；结尾可加「我的思考」。
