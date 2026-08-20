# Deep-Dream

Document-first concept graph knowledge server.

## 核心原则

**不要只信 CLI 的输出就下结论——用真实数据验证每一层。**

具体来说：
1. CLI 显示 0 不代表数据库是空的——先查实际的 DB 文件和表行数
2. 路径可能对不上——实际数据库文件是 `library/library.db`，不是 `graph.db`
3. 存储层 API 签名可能和 CLI 假设的不一致——先查实际方法签名再调用
4. 用户内容（source_text、content 等）可能包含 `[方括号]`，会被 Rich 误当 markup——必须 escape
5. 存储层可能返回 Entity DTO 而不是 dict——做 `.get()` 前先统一转 dict

## 自改进流程

当用户说 **"迭代优化系统"** 时，读取 `.claude/memory/self-improvement-loop.md`，用其中的 prompt 启动空上下文子 agent。

## 技术栈

- Python / Flask / SQLite
- CLI: Click 8+ / Rich 13+ (`core/cli/` package)
- Embedding: `core/llm/client.py` + `core/storage/embedding.py`
- Storage: `core/storage/sqlite/` (schema, manager, helpers)
- Pipeline: `core/remember/` (orchestrator, entity, relation, alignment)
- Web UI: `core/server/static/`
- 端口: `16200`

## 范围

`research/` 是实验测评（benchmark harness、数据、运行记录）与论文工作区，不属于 Deep-Dream 系统本体；改系统时不要动它，跑评测时进入它（用法见 `research/README.md`）。

## Skill

Deep-Dream 交互使用 `/deep-dream` skill，定义在 `.claude/skills/deep-dream/SKILL.md`。
