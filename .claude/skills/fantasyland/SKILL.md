---
name: fantasyland-novel
description: 基于 Deep-Dream 记忆图谱的长程叙事创作。适用于小说、故事、真实事件推演等需要长程一致性的叙事。核心流程：写前大量检索 → 写初稿 → 逐句多轮校验 → 全过才输出定稿并存入。彻底解决长篇创作"吃书"问题。需搭配 deep-dream 技能使用。
---

# Fantasyland — 长程叙事创作

解决一个核心问题：**长篇叙事的一致性**。

三阶段铁律：
1. **写前大量检索** — 把所有相关记忆翻出来
2. **写后逐句多轮校验** — 每句话去记忆库比对
3. **全过才输出存入** — 零冲突才输出定稿

## 适用场景

小说、故事、连载、真实事件推演、多线叙事、世界观设定集——任何需要长程一致的叙事。

## 工作空间

每次创作一个作品时，在工作目录下维护以下结构：

```
[作品名]/
├── deep_dream.json        # ★ Deep-Dream 连接配置（最先创建，每次存取前必读）
├── settings/              # 设定（写入后锁定，不可静默修改）
│   ├── world.md           # 世界观/环境/时代
│   ├── characters/        # 每个角色一个 md
│   │   ├── 角色名-1.md
│   │   └── 角色名-n.md
│   ├── systems.md         # 体系/规则/力量/阶级
│   └── factions.md        # 组织/势力/阵营
├── outline/               # 大纲
│   ├── master.md          # 总纲（整体弧线）
│   ├── vol1.md            # 卷纲
│   └── chapters/          # 章纲（每章一个 md）
│       └── ch01.md
├── foreshadowing.md       # 伏笔追踪表（编号/内容/埋设章/预计回收/状态）
├── chapters/              # 已完成章节定稿（只放通过的定稿）
│   ├── ch01.md
│   └── ch02.md
├── state/                 # 每章结束状态快照
│   ├── ch01_state.md
│   └── ch02_state.md
└── drafts/                # 当前写作暂存（写完清理）
    └── chXX_draft.md      # ← 章节通过后移到 chapters/，删除 draft
```

### deep_dream.json 格式

每个作品的**第一个文件**，新建作品时立刻创建：

```json
{
  "base_url": "http://localhost:16200/api/v1",
  "graph_id": "novel_作品英文名",
  "note": "所有 Deep-Dream 存取都基于此配置，不可遗漏"
}
```

### 文件管理规则

- `deep_dream.json` 每次子 Agent 调用前必须读取，拿到 `base_url` 和 `graph_id`
- `settings/` 写入后锁定，修改必须记录原因
- `settings/` 中任何文件被修改后，必须立即用子 Agent 将修改后的完整内容重新存入 Deep-Dream
- `outline/` 中章纲被修改后同理，同步存入
- `drafts/` 只放当前在写的初稿，通过后移到 `chapters/` 并删除 draft
- `state/` 每章写完后写入状态快照，下章写前必读
- `foreshadowing.md` 伏笔状态表，每章写前写后都要更新

### 写入验证

所有 `remember` 写入必须验证成功。子 Agent 指令中增加验证步骤：

```
# 存入时必须验证
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
使用 remember（wait=true, timeout=300）写入以下内容，source_name 为 '{source_name}'：
'{内容}'
写入后，使用 quick_search 用相同 source_name 查询，确认内容确实已存入。
如果查询不到，报告写入失败。"
```

`wait=true` 确保同步写入拿到结果，`quick_search` 二次确认内容确实进入图谱。只有验证成功才算写入完成。

## 子 Agent 调用

**核心原则：所有 Deep-Dream 记忆查询都通过子 Agent 执行。**

原因是子 Agent 上下文干净，只携带查询指令和 Deep-Dream skill，返回纯粹基于记忆的回答，不受主对话上下文干扰。

### 调用模式

使用 `Agent` 工具，`subagent_type="general-purpose"`，指令模板：

```
请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），执行以下查询：

{具体查询指令}

只返回查询结果，不要推理，不要发挥。
```

### 各阶段子 Agent 指令

#### 写前检索（批量查询，可并行多个子 Agent）

```
# 角色状态查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
查询以下角色的完整画像和最新状态：{角色名列表}。
使用 entity_profile 和 batch_profiles，返回每个角色的：
外貌特征、性格、当前能力、当前位置、身体状态、情绪、持有物品、近期经历。"

# 关系查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
查询 {角色A} 和 {角色B} 之间的关系。
使用 get_relations_between，返回：当前关系、过往事件、未说出口的事、对彼此态度。"

# 场景查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
查询 {地点名} 的完整信息。包括：地理描述、已知细节、相关事件、当前状态。"

# 伏笔查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
使用 quick_search 查询所有未回收的伏笔。返回每个伏笔的：
编号、内容、埋设章节、铺垫进度、预计回收章节、相关实体。"

# 规则查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
查询 {规则/体系名} 的完整设定。包括：等级划分、限制、代价、不可违反的约束。"

# 前文状态查询
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
使用 quick_search 查询 '状态-第{N}章结束' 的完整内容。
返回每个出场实体在上一章结束时的完整状态快照。"
```

#### 校验查询（每轮校验调用）

```
# 逐句校验
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
检查以下内容是否和记忆矛盾：'{待校验的句子}'
使用 quick_search 查询相关实体，逐项比对。只报告矛盾，不报告一致项。"

# 时间线校验
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
确认上一章结束的时间点是 {时间}。检查以下初稿的时间流逝是否合理：'{初稿}'"

# 信息泄露校验
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
检查 {角色名} 在以下文本中是否表现出他不应该知道的信息：'{文本}'
使用 quick_search 追溯该角色获取该信息的路径。如果无路径，报告为信息泄露。"
```

#### 记忆写入（通过校验后）

```
# 存入章节
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
使用 remember（wait=true）写入以下内容，source_name 为 '{source_name}'：
'{内容}'"

# 更新角色
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
对实体 {family_id} 执行 evolve_entity_summary，更新其摘要反映最新状态。"

# Dream 探索
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
执行 dream_run，strategy='{strategy}'，seed_count=3，返回发现的新关联。"
```

## 创作流程

### 阶段 A：设定构建

1. 从用户描述中提炼核心要素
2. 分批写入设定（角色→世界观→体系→组织），每批用子 Agent 调用 `remember` 存入
3. 写入后用子 Agent 验证：`quick_search` 确认抽取完整，`create_entity`/`create_relation` 补全遗漏
4. 设定同步写入 `settings/` 目录下的对应 md 文件
5. 用子 Agent 执行 `dream_run(strategy="cross_domain")` 发现隐藏关联

### 阶段 B：大纲规划

1. 用子 Agent 检索所有已写入设定
2. 生成总纲→卷纲→章纲，每层用子 Agent 调用 `remember` 存入
3. 章纲同步写入 `outline/chapters/`
4. 初始化 `foreshadowing.md`
5. 用子 Agent 执行 `dream_run(strategy="narrative")`

### 阶段 C：逐章写作

对每一章执行以下三步循环：

#### Step 1：写前大量检索

并行启动多个子 Agent（至少 5 个）：

| 子 Agent | 查询内容 |
|----------|---------|
| 1 | 本章大纲 (`quick_search` 大纲-第X章) |
| 2 | 上一章状态 (`quick_search` 状态-第X-1章结束) |
| 3 | 出场角色画像 (`batch_profiles` + `entity_profile`) |
| 4 | 角色间关系 (`get_relations_between` 每对) |
| 5 | 未回收伏笔 (`quick_search` 伏笔) |
| 6 | 场景/规则信息 (`quick_search`) |

汇总结果后，生成章节场景预设，用子 Agent 存入记忆，写入 `drafts/chXX_draft.md`。

#### Step 2：写初稿

基于检索到的全部记忆上下文写初稿。初稿**不展示给用户**，不存入记忆。

#### Step 3：逐句多轮迭代校验

对初稿执行五轮校验，任何一轮发现冲突→修改→从第1轮重来：

| 轮次 | 校验内容 | 方法 |
|------|---------|------|
| 1 | 逐句实体校验 | 每句话用子 Agent `quick_search` 查询比对 |
| 2 | 时间线连续性 | 子 Agent 确认时间衔接和流逝 |
| 3 | 信息泄露 | 子 Agent 追溯每个角色的信息获取路径 |
| 4 | 伏笔/因果 | 子 Agent 查伏笔状态，检查因果链 |
| 5 | 全局通读 | 子 Agent 整章通读：逻辑、AI味、节奏 |

★ 五轮全部零冲突 → 通过

#### Step 4：输出与存入（仅通过后）

1. 输出定稿给用户
2. 初稿从 `drafts/` 移到 `chapters/chXX.md`，删除 draft
3. **串行**执行以下子 Agent（必须等前一个完成再启动下一个）：
   - `remember(wait=true, timeout=300)` 存入完整章节 → `quick_search` 验证
   - `remember(wait=true, timeout=300)` 存入状态变更 → `quick_search` 验证
   - `evolve_entity_summary` 更新每个出场角色
   - `dream_run(strategy="narrative")` 发现新关联
4. 更新 `foreshadowing.md`

★ **章节存入全部完成后，再开始写下一章。** 可用子 Agent 执行验证：

```
"请使用 deep-dream 技能，访问记忆库（graph_id: "{graph_id}"），
使用 quick_search 查询以下内容，确认全部存在：
1. source_name='正文-第X章-xxx'
2. source_name='状态-第X章结束'
3. 出场角色 {角色名列表} 的最新状态
如果有任何一项查询不到，报告缺失项，等待补充存入。"
```

## 写作质量规则

### 对白

每句对话带三样：关系位置、信息推进、个性。不用旁白解释对话含义。

### 章末

收在变化上（真相一角/新危机/被迫选择/关系变化），不收在总结句。

### 去 AI 味

- 能用动作不用总结（"她放下杯子" 优于 "她内心复杂"）
- 能用对白不用解释
- 能写具体不写抽象
- 能断句不拖长句
- 每人说话带个性，不堆套词

## 硬约束

- **不跳过写前检索直接写**
- **不写完就输出，必须五轮校验**
- **校验未通过不存入记忆**
- **不静默修改已锁定设定**
- **上一章未完全存入前，不开始写下一章**
- **所有 Deep-Dream 操作通过子 Agent 执行**
