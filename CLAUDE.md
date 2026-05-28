# Deep-Dream

Document-first concept graph knowledge server.

## 自改进流程

当用户说 **"迭代优化系统"** 时，读取 `.claude/memory/self-improvement-loop.md`，用其中的 prompt 启动空上下文子 agent。

## 技术栈

- Python / Flask / SQLite
- Embedding: `core/llm/client.py` + `core/storage/embedding.py`
- Storage: `core/storage/sqlite/` (schema, manager, helpers)
- Pipeline: `core/remember/` (orchestrator, entity, relation, alignment)
- Web UI: `core/server/static/`
- 端口: `16200`

## Skill

Deep-Dream 交互使用 `/deep-dream` skill，定义在 `.claude/skills/deep-dream/SKILL.md`。
