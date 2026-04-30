---
name: fantasyland-novel
description: 基于 Deep-Dream 记忆图谱的长程叙事创作。核心：写前检索→写初稿→逐句校验→全过才输出。彻底解决"吃书"。需搭配 deep-dream 技能。v3 — 加入 remember 完成等待协议和阶段门控。
---

# Fantasyland — 长程叙事创作 v3

解决一个核心问题：**长篇叙事的一致性**。

三阶段铁律：
1. **写前大量检索** — 把所有相关记忆翻出来
2. **写后逐句校验** — 每个关键细节去记忆库比对
3. **全过才输出存入** — 零冲突才输出定稿

> **v3 变更说明**：基于4章实测发现图谱严重落后于写作的致命问题。核心改进：remember 完成等待协议（阶段门控）、写入内容精炼规则、全部统一用 remember 不绕路。

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
├── foreshadowing.md       # 伏笔追踪表
├── chapters/              # 已完成章节定稿（只放通过的定稿）
│   ├── ch01.md
│   └── ch02.md
├── state/                 # 每章结束状态快照（YAML格式，便于解析）
│   ├── ch01_state.yaml
│   └── ch02_state.yaml
└── drafts/                # 当前写作暂存（写完清理）
    └── chXX_draft.md
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
- `state/` 每章写完后写入状态快照（YAML格式），下章写前必读
- `foreshadowing.md` 伏笔状态表，每章写前写后都要更新

## API 调用备忘

> **重要**：Deep-Dream 的检索端点全部是 **POST**，不是 GET。

### 正确的 API 调用方式

```bash
# quick_search — POST + JSON body
curl -s -X POST "$BASE_URL/find/quick-search" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"$GRAPH_ID","query":"搜索内容"}'

# remember — POST + JSON body（注意：每条约10分钟，慎用）
curl -s -X POST "$BASE_URL/remember" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"$GRAPH_ID","text":"内容","source_name":"前缀-名称","wait":true}'

# create_entity — 直接创建实体（比 remember 快很多）
curl -s -X POST "$BASE_URL/entities?graph_id=$GRAPH_ID" \
  -H "Content-Type: application/json" \
  -d '{"name":"实体名","entity_type":"角色|地点|物品|事件","summary":"描述"}'

# entity_profile — 获取实体完整画像
curl -s "$BASE_URL/entities/$FAMILY_ID/profile?graph_id=$GRAPH_ID"

# batch_profiles — 批量获取实体画像（最多20个）
curl -s -X POST "$BASE_URL/entities/batch-profiles?graph_id=$GRAPH_ID" \
  -H "Content-Type: application/json" \
  -d '{"family_ids":["ent_xxx","ent_yyy"]}'

# find_entity_by_name — 按名称查找
curl -s "$BASE_URL/find/entity-by-name?graph_id=$GRAPH_ID&name=苏婉"

# search_concepts — 跨角色统一搜索
curl -s -X POST "$BASE_URL/find/concepts?graph_id=$GRAPH_ID" \
  -H "Content-Type: application/json" \
  -d '{"query":"搜索内容","search_mode":"hybrid"}'
```

### remember vs 直接操作

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 初始设定写入 | `remember(wait=true)` | 需要自动抽取实体和关系 |
| 章节正文存入 | `remember(wait=true)` | 需要自动抽取事件和状态变更 |
| 状态快照更新 | `create_entity` + `create_relation` | 直接操作，速度快 |
| 角色属性修正 | `update_entity` | 精准修改，不触发全量抽取 |
| 角色摘要更新 | `evolve_entity_summary` | 专门的摘要演进工具 |

### 写入验证

所有 `remember` 写入必须验证成功：

```bash
# 写入后验证
TASK_ID=$(curl -s -X POST "$BASE_URL/remember" -H "Content-Type: application/json" \
  -d '{"graph_id":"$GRAPH_ID","text":"内容","source_name":"名称","wait":true}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('task_id',''))")

# 轮询直到完成
while true; do
  STATUS=$(curl -s "$BASE_URL/remember/tasks/$TASK_ID?graph_id=$GRAPH_ID" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(d['status'])")
  [ "$STATUS" = "completed" ] && break
  sleep 30
done

# 二次确认：用 quick_search 查询
curl -s -X POST "$BASE_URL/find/quick-search" -H "Content-Type: application/json" \
  -d '{"graph_id":"$GRAPH_ID","query":"刚写入的source_name关键词"}'
```

## 子 Agent 调用

**核心原则：所有 Deep-Dream 记忆查询都通过子 Agent 执行。**

子 Agent 上下文干净，不受主对话上下文干扰。但**前文定稿文件**必须作为上下文传入。

### 调用模式

```
使用 Agent 工具，subagent_type="general-purpose"

关键：在 prompt 中同时提供：
1. Deep-Dream 连接信息（graph_id, base_url）
2. 需要读取的文件路径（前章定稿、状态快照）
3. 具体的 API 调用命令模板
4. 一致性检查清单（具体到每个细节）
```

### 写作子 Agent Prompt 模板

```markdown
你是小说写作 agent。按 fantasyland 技能流程写第X章《标题》。

## 配置
- Deep-Dream: graph_id={graph_id}, base_url={base_url}
- 工作目录: {work_dir}

## Step 1: 写前检索
并行执行以下 curl 调用（全部是 POST）：
1. quick_search: "{本章大纲关键词}"
2. quick_search: "{角色A关键词}" — 获取完整画像
3. quick_search: "{角色B关键词}" — 获取完整画像
4. quick_search: "{场景/地点关键词}"
5. quick_search: "{规则/物品关键词}"
同时读取前章定稿文件: {前章文件路径}
同时读取前章状态快照: {前章状态文件路径}

## Step 2: 写初稿
- 字数目标: 1500-2500字（不得超过3000字）
- 场景序列: {场景列表}
- 时间线: {起止时间}

## Step 3: 一致性校验（逐项检查）
以下细节必须正确（错误任何一个都要改）：
- {角色A}: {具体属性列表，如：左眼失明、右手虎口疤痕、短发}
- {角色B}: {具体属性列表}
- {物品/地点}: {具体属性}
- {时间规则}: {具体时间点}
- {前章状态衔接}: {关键状态}

## Step 4: 输出
定稿写入: {输出文件路径}
状态快照写入: {状态文件路径}

## Step 5: 存入验证（必须通过）
1. remember(wait=true) 存入章节正文（source_name: "正文-第X章-标题"）
2. 轮询验证完成（GET /remember/tasks/{task_id}，直到 status=completed）
3. quick_search 抽查确认内容可检索
4. evolve_entity_summary 更新每个出场角色
5. 全部验证通过后返回成功
```

## 创作流程

### `remember` 完成等待协议（★ 关键 — 防止图谱落后于写作）

> **铁律**：全部用 `remember(wait=true)` 写入。每条 `remember` 约10分钟，这是正常的。安心等它跑完，不要用 `create_entity` 等直接 API 绕路——只有 `remember` 才能自动抽取实体、关系、事件，才能保证图谱的丰富性和检索质量。
>
> **内容质量要求**：传入 `remember` 的文本必须千锤百炼——信息密度高、表述精准、无冗余。好的输入 = 好的抽取 = 好的检索。不要把粗糙的草稿塞进记忆库。

#### 阶段门控（Phase Gate）

```
阶段 A（设定）全部 remember 完成 → 才能进入阶段 B
阶段 B（大纲）全部 remember 完成 → 才能进入阶段 C
阶段 C 每章的 remember 完成 → 才能写下一章
```

每一条 `remember` 都必须等到 `status=completed` 后，才能提交下一条（因为 remember 是串行处理的，排队会延长总等待）。验证方法：

```bash
# 轮询单条任务直到完成
curl -s "$BASE_URL/remember/tasks/$TASK_ID?graph_id=$GRAPH_ID" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']
print(f'{d[\"source_name\"]}: {d[\"status\"]} ({d.get(\"progress\",0):.0%})')
"

# 全局检查：是否还有未完成的任务
curl -s "$BASE_URL/remember/tasks?graph_id=$GRAPH_ID" | python3 -c "
import sys, json
tasks = json.load(sys.stdin)['data']['tasks']
pending = [t for t in tasks if t['status'] in ('running', 'queued')]
if pending:
    print(f'BLOCKED: {len(pending)} tasks pending')
    for t in pending:
        print(f'  - {t.get(\"source_name\",\"?\")}: {t[\"status\"]}')
else:
    print('ALL COMPLETE')
"
```

#### 写入内容精炼规则

传入 `remember` 的文本必须满足：

1. **信息密度**：每句话都包含可抽取的实体/关系/状态。不用"在这座美丽的岛上"这种空洞描述。
2. **命名一致**：角色名、地名、物品名全篇统一，不用别名或简称。
3. **状态明确**：角色的位置、情绪、身体状态、持有物品、已知信息——全部写清楚。
4. **关系显式**：角色之间的关系用自然语言明确写出（"陆远对苏婉持怀疑态度"而非暗示）。
5. **时间锚点**：每个事件附带明确时间（"1952年10月15日17:30"）。

### 阶段 A：设定构建

1. 从用户描述中提炼核心要素
2. **精炼**设定文本，确保满足写入内容精炼规则
3. 用 `remember(wait=true)` 存入，**等完成再提交下一条**
4. 每条完成后 `quick_search` 抽查验证可检索
5. 设定同步写入 `settings/` 目录下的对应 md 文件
6. 全部设定 remember 完成 → 进入阶段 B
7. 可选：`dream_run(strategy="cross_domain")` 发现隐藏关联

### 阶段 B：大纲规划

1. 检索所有已写入设定（必须先确认阶段 A 全部完成）
2. 生成总纲→卷纲→章纲
3. **精炼**大纲文本，确保满足写入内容精炼规则
4. 用 `remember(wait=true)` 存入，**等完成再提交下一条**
5. 每条完成后 `quick_search` 抽查验证
6. 章纲同步写入 `outline/chapters/`
7. 初始化 `foreshadowing.md`
8. 全部大纲 remember 完成 → 进入阶段 C

### 阶段 C：逐章写作（核心循环）

```
┌─────────────────────────────────────────────────────────────┐
│                  章节循环 (Chapter Loop)                      │
│                                                              │
│  ① 写前检索（并行）                                           │
│     - 本章大纲                                                │
│     - 前章状态快照 + 前章定稿                                  │
│     - 出场角色完整画像                                         │
│     - 场景/规则/物品                                           │
│     - 未回收伏笔                                              │
│                                                              │
│  ② 写初稿（1500-2500字）                                      │
│                                                              │
│  ③ 一致性校验（结构化清单）                                     │
│     - 角色属性校验                                             │
│     - 状态连续性校验                                           │
│     - 时间线校验                                              │
│     - 信息边界校验                                             │
│     - 伏笔状态校验                                             │
│     发现矛盾 → 修改 → 重新校验                                 │
│                                                              │
│  ④ 全过 → 输出定稿                                            │
│                                                              │
│  ⑤ 存入记忆 + 状态快照                                        │
│     - remember 存入章节正文                                    │
│     - 状态快照写入 YAML 文件                                   │
│     - evolve_entity_summary 更新角色                          │
│                                                              │
│  → 进入下一章                                                 │
└─────────────────────────────────────────────────────────────┘
```

#### Step 1：写前大量检索

并行检索（至少5项），**同时读取前章定稿文件和状态快照文件**：

| 检索项 | API 调用 |
|--------|---------|
| 本章大纲 | `quick_search(query="大纲-第X章")` |
| 前章定稿 | 直接读取 `chapters/ch(X-1).md` 文件 |
| 前章状态 | 直接读取 `state/ch(X-1)_state.yaml` 文件 |
| 出场角色画像 | `quick_search` + `entity_profile` |
| 角色间关系 | `get_relations_between` 每对 |
| 未回收伏笔 | `quick_search(query="伏笔 未回收")` |
| 场景/规则 | `quick_search(query="场景关键词")` |

**隐藏细节发现**：对每个出场角色，额外调用 `entity_profile` 获取完整属性列表，确保不遗漏（如玻璃义眼这类 quick_search 可能漏掉的细节）。

**缺失实体处理**：如果某个角色/地点在图谱中查不到：
1. 先检查 settings/ 目录下的 md 文件是否包含该信息
2. 如果文件中有 → 先用 `create_entity` 补入图谱，再继续写作
3. 如果文件中也没有 → 从大纲中提取并确认后补入

#### Step 2：写初稿

基于检索到的全部上下文写初稿。

**字数控制**：
- 目标：1500-2500字
- 硬上限：3000字
- 超过3000字必须删减次要场景
- 不足1500字需要补充细节

#### Step 3：结构化一致性校验

使用以下清单逐项校验。**任何一项不通过则修改初稿并重新校验**。

##### 3.1 角色属性校验

对每个出场角色，检查以下维度：

```
角色: {角色名}
□ 外貌：{具体外貌特征列表，如"左眼失明、短发到耳根、右手虎口疤痕"}
□ 性格：{性格关键词}
□ 能力/弱点：{能力列表}
□ 持续性细节：{跨章必须保持的微小细节，如"陆远从不点燃香烟"}
□ 当前状态：{位置、情绪、身体状态}
```

##### 3.2 状态连续性校验

```
时间线: 第X章 → 第X+1章
□ 前章结束状态 → 本章开始状态衔接正确
□ 受伤/疲劳等负面状态持续（不会自动消失）
□ 物品持有连续（不会凭空出现或消失）
□ 位置移动合理（不能瞬移）
□ 情绪变化有因果（不能无理由翻转）
```

##### 3.3 时间线校验

```
□ 潮汐/日出日落/月相等自然时间正确
□ 角色行动时间与地点距离匹配
□ 跨天情节日期正确
□ 季节/天气描述一致
```

##### 3.4 信息边界校验

```
□ 角色不知道他不该知道的信息（不能读心）
□ 信息获取有合理路径（追溯信息来源）
□ 已知信息不会"被遗忘"（角色应该记得的事）
```

##### 3.5 伏笔/因果校验

```
□ 未回收伏笔持续存在（不能遗忘）
□ 新伏笔不与已有伏笔矛盾
□ 因果链合理（行为有动机）
```

#### Step 4：输出与存入

1. 输出定稿给用户
2. 初稿从 `drafts/` 移到 `chapters/chXX.md`，删除 draft
3. 状态快照写入 `state/chXX_state.yaml`
4. 执行存入：
   - `remember(wait=true)` 存入完整章节（source_name: `正文-第X章-标题`）
   - **轮询验证 remember 完成**（参考「完成验证脚本」）
   - `evolve_entity_summary` 更新每个出场角色
5. 更新 `foreshadowing.md`
6. **全部存入验证通过后，才能开始下一章**（阶段门控）

#### 跨章衔接协议（章节间必须执行）

```
上一章完成后，下一章开始前：
1. 读取上一章定稿文件（全文）
2. 读取上一章状态快照（YAML）
3. 用 quick_search 确认关键实体状态在图谱中正确
4. 对比文件中的状态和图谱中的状态是否一致
5. 不一致时以文件为准（因为 remember 可能有延迟）
```

### 状态快照格式（YAML）

```yaml
chapter: 1
title: "来客"
time: "1952-10-15 20:00"
weather: "大雾，能见度不足50米"
location_weather: "深秋，大西洋冷风"

characters:
  苏婉:
    location: "管理员平房，自己卧室"
    status: "疲惫，侧躺"
    mood: "平静但警觉"
    body: "右眼正常，左眼失明（玻璃义眼），右手虎口烧伤疤痕"
    items: []
    knowledge:
      - "知道陆远是大陆来的调查员"
      - "不知道陆远的真实调查目的"

  陆远:
    location: "管理员平房，客用卧室"
    status: "安顿中"
    mood: "公事公办，隐藏目的"
    body: "右腿膝盖旧伤，平地微跛"
    items: ["灰色风衣", "密封档案（牛皮纸信封，火漆印）", "香烟"]
    knowledge:
      - "知道苏婉是灯塔管理员"
      - "真实目的是调查苏婉父亲死因"

  老郑:
    location: "已返回大陆"
    status: "不在岛上"

facilities:
  灯塔:
    status: "已点亮"
    lit_at: "17:30"
    flash_interval: "12秒"
    fuel_remaining: "约7天"
  码头:
    status: "涨潮中"
    last_boards_submerged: true
  电话:
    status: "预计断线（大雾）"

supplies:
  面粉: "半袋（约够10-14天）"
  煤油: "一桶"
  腌菜: "几罐"
  盐: "一小袋"
  蜡烛: "快用完"

foreshadowing_planted:
  - id: "001"
    content: "老郑'知道得越少越好'"
    chapter: 1
    status: "planted"
  - id: "002"
    content: "苏婉父亲照片被翻过来扣着"
    chapter: 1
    status: "planted"
  - id: "003"
    content: "煤油储量低于正常水平"
    chapter: 1
    status: "planted"

notes: "陆远从不在岛上点燃香烟（跨章细节）"
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

### 跨章微小细节

识别并记录跨章必须保持的微小细节（如"陆远从不点燃香烟"），写入状态快照的 `notes` 字段。这些细节是"吃书"最容易暴露的地方。

## 硬约束

- **不跳过写前检索直接写**
- **不写完就输出，必须完成结构化校验清单**
- **校验未通过不存入记忆**
- **不静默修改已锁定设定**
- **所有内容一律用 remember(wait=true) 存入，等完成再继续**（阶段门控）
- **传入 remember 的文本必须精炼——信息密度高、命名一致、状态明确、关系显式、时间锚点**
- **上一章 remember 未完成前，不开始写下一章**
- **所有 Deep-Dream 操作通过子 Agent 执行**
- **每章字数不超过3000字**
- **API 调用全部用 POST（不是 GET）**
- **写作前必须读取前章定稿全文，不能只靠图谱**
