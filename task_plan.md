# Task Plan: Deep-Dream 系统全面开发与优化

## Goal
基于 Deep-Dream-CLI.md 设计文档，对 Deep-Dream 知识图谱系统进行前端优化、后端完善、测试覆盖、性能调优的全面开发工作，使其成为生产可用的 Agent 记忆系统。

## Current Phase
Phase 5b: 提取质量优化（挑战式Prompt + think=false）— ✅ 完成

## Phases

### Phase 1: 现状审计与需求确认 ✅
- **Status:** complete

### Phase 2: 后端 API 完善与稳健性 ✅
- **Status:** complete

### Phase 3: 前端优化与用户体验
- **Status:** pending (deferred to user-directed iteration)

### Phase 4: 测试覆盖与质量保障 (部分完成)
- **Status:** in_progress

### Phase 5: 性能调优 — 步骤6实体处理优化 ✅

### Phase 5b: 提取质量优化（挑战式Prompt + think=false）✅

#### 背景
发现步骤5关系发现（189-234s）是新的瓶颈。测试 think=false vs think=true 发现：
- 初始抽取质量相同（18对 vs 18对）
- 但 think=false 的 refinement 找 0 对新关系，think=true 找 9 对
- 最初尝试 hybrid（初始 false + 精炼 true），但用户建议试试改进 prompt

#### 关键发现：Ollama API 差异
- **OpenAI 兼容 API (`/v1/chat/completions`)**：think=false 时模型仍大量推理，占满 token 预算
- **Ollama 原生 API (`/api/chat`)**：think=false 正确抑制推理，8s 完成初始抽取
- 服务器使用原生 API，所以 think=false 正常工作

#### Prompt A/B/C/D/E 对比测试（原生 API）
| 变体 | think | R1耗时 | 新增关系对 | 备注 |
|------|-------|--------|-----------|------|
| A 原始 | false | 7.5s | +1 | 过于保守 |
| **B 挑战式** | **false** | **7.3s** | **+4** | **最多** |
| C 结构化 | false | 12.1s | +2 | 中等 |
| D 断言式 | false | 6.2s | ~some | 快 |
| E 混合 | true | 89.6s | +2 | 15x慢，反而更少 |

**结论：think=true 反而更差**（模型把 token 花在推理上，实际输出更少）
**B（挑战式）效果最好** — "你找的还不够全面"直接刺激模型输出更多

#### 抽取质量验证
对照设计文档 "第二条：提取最大化" 和 "第三条：实体 = 记忆锚点"：

**实体（W1: 95→73 唯一）：**
- 约 93% 符合 "记忆锚点" 标准
- 覆盖：核心人物、地点、金句名句、核心思想、标志性场景、新奇概念、人物特质
- 之前的 ~20 个严重漏提取，现在 73 个大幅覆盖

**关系（W1: 71→141 对）：**
- 每条关系都有具体内容（非泛泛描述）
- 覆盖：因果关系、家庭关系、地理关系、事件关系
- 之前的 27 对严重漏提取

**设计文档核心理念完全印证：**
> "越多越好。漏提取比多提取危害大得多：多的可过滤，漏的永远找不回来"

#### 已实施变更
- [x] RELATION_REFINE_USER 改为挑战式："你找的关系还不够全面..."
- [x] ENTITY_REFINE_USER 改为挑战式："你提取的概念还不够全面..."
- [x] 移除 hybrid think=true 逻辑（refine_think_mode 参数）
- [x] 移除 client.py 中的 think_override 机制

#### Pipeline 时间对比
| 指标 | opt3 (原始prompt) | challenge prompt | 变化 |
|------|-------------------|------------------|------|
| 步骤1 实体提取 | ~30s | 47.9s | +60% |
| 步骤5 关系发现 | ~90s | 252.8s | +2.8x |
| 总流水线 | 310.8s | 1843.9s | +5.9x |

**时间增加原因：抽取量大幅增加**
- 实体：~20→73（3.7x）
- 关系：~27→150+（5.6x）
- 下游步骤（内容写作、实体对齐、关系对齐）负载相应增加
- 下一步：优化下游步骤处理更大数据量

**Status:** complete

#### 优化结果对比
| 指标 | 基准 (opt2) | 优化后 (opt3) | 改善 |
|---|---|---|---|
| 步骤6 实体处理 | 403.4s | **30.9s** | **13x** |
| 步骤6 总计 | 606.5s | **51.0s** | **11.9x** |
| 每实体均耗时 | ~40s | **~1.5s** | **26x** |
| 步骤7 关系对齐 | 6.8s | 4.3s | 1.6x |
| 总流水线耗时 | 994.8s | **310.8s** | **3.2x** |

#### 已实施优化
- [x] 5.1 Trust "create_new" at confidence >= 0.5
- [x] 5.4 限制 detailed analysis 候选数到 top 5
- [x] 5.5 Detailed analysis 并行度 2→3 workers
- [x] 5.10 Skip _alignment_guard for merge_safe exact/substring matches (exact_match_fast)
- [x] 5.11 Lightweight path for moderate confidence (0.5-0.75) merge decisions
- [x] 5.12 Skip preliminary analysis in sequential fallback
- [x] **5.13 Enable alignment config with think=false** → 最大收益：步骤6/7的LLM调用不再生成thinking tokens
- [x] **5.14 Content overlap fast path in _merge_two_contents** → bigram Jaccard >= 0.55 时跳过LLM合并

#### 验证：对齐质量未降低
- 合并实体（如 百无聊赖+独自垂泪）内容正确整合了新旧信息
- 关系正确保留和扩展

**Status:** complete

### Phase 5c: 下游步骤优化（内容写作 Fast-path）✅

#### 背景
挑战式 prompt 提升了抽取量（实体3.7x，关系5.6x），但下游步骤（内容写作、对齐）时间大幅增加：
- 步骤3 实体内容写作：346s（73个实体）
- 步骤6 关系内容写作：340s（141+对关系）
- 步骤6 实体对齐：532s（73个实体 vs 已有图谱）

所有 LLM 调用通过单个 RTX 3090 GPU 串行执行，因此并行化无效，关键策略是**减少总 LLM 调用次数**。

#### 核心优化：窗口文本 Fast-path
利用窗口文本中的句子作为内容，跳过 LLM 生成：

**实体内容 fast-path（Step 3）：**
- 预计算 `_build_entity_fallback_content`（基于 bigram 索引查找包含实体名的句子）
- 如果回退内容 >= 15字 且通过质量门 → 直接使用，跳过 LLM
- 效果：W1 命中率 97%（61/63），W2 命中率 85%（17/20）

**关系内容 fast-path（Step 6）：**
- 新增 `_build_relation_fallback_content`：查找同时包含两个实体名的句子
- 如果找到共现句子 → 直接作为关系内容
- 效果：W1 命中率 60%（37/62），W2 命中率 65%（24/37）

**批处理块大小增加：**
- `batch_write_entity_content` chunk_size: 20→35
- `batch_write_relation_content` chunk_size: 20→35

#### 优化结果（dd_server11 vs dd_server9 基准）
| 步骤 | 基准 (dd_server9) | 优化后 (dd_server11) | 改善 |
|------|-------------------|----------------------|------|
| 步骤3 实体内容 | 346.1s | **8.9s** | **39x** |
| 步骤6 关系内容 | 339.9s | **41.8s** | **8x** |
| 步骤2-5 抽取 | 986.8s (3窗) | **151.2s** (2窗) | **~6.5x/窗** |
| 步骤6 实体对齐 | 614.7s (3窗) | **85.6s** (2窗) | **~3.5x/窗** |
| 总流水线 | 1843.9s (3窗) | **326.4s** (2窗) | **~4.7x/窗** |

**注意**：dd_server11 仅 2 窗口（W1 步骤7 有预存 bug），基准 dd_server9 为 3 窗口。每窗口对比更有参考价值。

#### 已实施变更
- [x] 新增 `_build_relation_fallback_content` 函数
- [x] Step 3: 实体内容 fast-path（先计算回退内容，好的直接用，不调 LLM）
- [x] Step 6: 关系内容 fast-path（共现句子直接作为关系内容）
- [x] batch chunk_size 20→35

#### 质量评估
- 实体回退内容是原文句子拼接，比 LLM 生成的更精确（对齐时是实际文本证据）
- 关系回退内容是同时包含两个实体的原文句子，包含具体关联上下文
- 未通过质量门的实体仍走 LLM 路径，保证底线质量

**Status:** complete

### Phase 6: 文档与交付
- **Status:** pending

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| refresh_relates_to_edges TypeError | dd_server11 W1 | 预存 bug，与优化无关。core/remember/relation.py:231 调用 self.storage.refresh_relates_to_edges |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5c 完成：下游步骤优化 39x/8x 加速 |
| Where am I going? | 可选：进一步优化实体对齐（85.6s→?），或进入 Phase 6 |
| What's the goal? | 保持高覆盖度抽取，总时间控制在合理范围 |
| What have I learned? | 窗口文本回退是零成本高质量内容源；85-97% 的实体不需要 LLM 写内容；GPU 串行化使减少调用比并行化更重要 |
| What have I done? | 实体/关系内容 fast-path、批处理块增大、验证测试通过 |
