# 示例：阅读前预存（即将开始阅读）

当你获取到 PDF / 网页 / 文件准备阅读时，**先把完整内容存入记忆**，再开始分析。这样记忆与真实时间线同步，忠实记录「此时开始读这份文档」。

## 查询参数（GET /api/remember）

正文较长时用 `text_b64`；下面示意用 `curl --data-urlencode` 传 `text`：

```bash
curl -s -G "http://127.0.0.1:16200/api/remember" \
  --data-urlencode "text=[2026-03-10 09:15] Kael — 准备阅读

用户分享了一篇 PDF《Attention Is All You Need》……
---
{PDF 提取的全部文字}
---" \
  --data-urlencode "source_name=论文-Attention" \
  --data-urlencode "event_time=2026-03-10T09:15:00"
```

要点：时间用「开始阅读」的时刻；前缀说明文档来源与动作；`---` 内是完整原文。读完后可再发一条带总结的 remember（参考 read-document.md）。
