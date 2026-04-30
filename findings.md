# Findings & Decisions — Deep-Dream 系统开发

## Requirements
- 遵循 Deep-Dream-CLI.md 设计哲学："万物皆概念"、版本化管理、溯源完整性
- 后端：API 完善、错误处理、Dream Agent 优化、任务队列稳健性
- 前端：SPA 优化、图谱可视化、搜索体验、Chat 流式输出、响应式
- 测试：API 集成测试、Remember 端到端测试、存储层测试、前端测试
- 性能：Neo4j 查询优化、LLM 调用并行化、Embedding 批处理、缓存调优

## Research Findings

### 项目架构
- **后端**: Flask (Python), 多图谱隔离 (GraphRegistry), 7步 Remember 流程
- **存储**: Neo4j (主) / SQLite (备), sqlite-vec 向量检索
- **LLM**: Ollama 本地部署 (Gemma4-26b-32k), 支持双模型管线 (extraction + alignment)
- **前端**: 单页应用, vis-network 图谱可视化, 多页面 (dashboard/chat/graph/search/entities/relations/episodes/dream)
- **测试**: pytest, conftest.py 提供基础 fixtures, 覆盖率低

### 已有性能分析 (2份报告)
1. **performance_analysis_report.md** (2026-04-27): 识别 5 个瓶颈, 已实施 Step6/Step7 并行度提升 + Embedding 缓存配置化
2. **performance_analysis_2026_04_26.md**: 识别 12 个问题, Phase 1 快速优化 (复合索引/缓存TTL/信号量) 待实施

### 已识别的性能瓶颈 (按优先级)
| 优先级 | 问题 | 状态 | 预估提升 |
|--------|------|------|----------|
| HIGH | Step6 实体对齐顺序处理 | ✅ 已修复并行度 | 20-30% step6 |
| HIGH | 复合索引缺失 (graph_id, family_id, invalid_at) | 待实施 | 20-30% 查询 |
| HIGH | Batch LLM 对齐调用 | 待实施 | 30-40% step6 |
| MEDIUM | resolve_family_id 缓存 TTL 过短 (120s) | 待实施 | 减少DB往返 |
| MEDIUM | Embedding 信号量硬编码=2 | 待实施 | 100% embedding吞吐 |
| MEDIUM | Neo4j N+1 查询模式 | 待实施 | 20-50% 查询 |
| LOW | Embedding 缓存无上限 | 待实施 | 30% 内存 |
| LOW | 全文索引配置优化 | 待实施 | 低影响 |

### 前端已有改进
- 时间色编码 (冷色→暖色)
- 概念类型边框样式 (实体=实线, 关系=虚线, 观察=点线)
- 搜索高亮与分数徽章
- 仪表盘实时刷新 (各区域独立刷新周期)
- 时间轴滑块 (已实现)

### 安全审计 (SECURITY_AUDIT_REPORT.md)
- 待审查具体内容

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 保持"万物皆概念"模型 | CLI 文档核心设计, 所有三种角色(Entity/Relation/Observation)统一为 Concept |
| 性能优化先做低风险项 | 复合索引/缓存TTL/信号量调优改动小、风险低、收益明确 |
| 测试优先覆盖 API 端点 | API 是系统入口, 最关键的质量保障点 |
| 保持双后端兼容 (Neo4j/SQLite) | SQLite 用于开发和测试, Neo4j 用于生产 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| (暂无) | - |

## Resources
- 设计文档: `/home/linkco/exa/Deep-Dream/Deep-Dream-CLI.md`
- 性能报告 1: `/home/linkco/exa/Deep-Dream/docs/performance_analysis_report.md`
- 性能报告 2: `/home/linkco/exa/Deep-Dream/docs/performance_analysis_2026_04_26.md`
- 前端改进: `/home/linkco/exa/Deep-Dream/frontend_improvements_summary.md`
- 开发文档: `/home/linkco/exa/Deep-Dream/core/docs/DEVELOPMENT.md`
- 安全审计: `/home/linkco/exa/Deep-Dream/SECURITY_AUDIT_REPORT.md`
- 后端入口: `core/server/api.py` (Flask app factory)
- 记忆管线: `core/remember/orchestrator.py` (7步流程)
- 存储层: `core/storage/` (SQLite + Neo4j)
- 前端入口: `core/server/static/index.html`
- 测试: `core/tests/`

---
*Update this file after every 2 view/browser/search operations*
