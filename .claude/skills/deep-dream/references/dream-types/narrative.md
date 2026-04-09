# 叙事编织 (Narrative)

## 策略
将碎片化的实体和关系编织成连贯的叙事，发现叙事中缺失的环节。

## 执行步骤

### 1. 选择叙事种子
```
get_dream_seeds(strategy=hub, count=1)  ← 一个核心实体作为叙事中心
get_dream_seeds(strategy=random, count=2) ← 辅助视角

核心实体 = hub 实体（关系丰富，有故事可讲）
```

### 2. 构建叙事骨架
```
core = get_entity(family_id=hub)
core_relations = get_entity_relations(family_id=hub, direction=both)

# 按时间排列关系（如果有时间信息）
timeline = get_entity_timeline(family_id=hub, limit=20)

# 构建叙事元素：
  人物（Person 类型邻居）
  事件（Event 类型邻居或 timeline 事件）
  概念（Concept 类型邻居）
  组织（Organization 类型邻居）
```

### 3. 发现叙事间隙
```
对叙事链中的相邻元素 (A, B):
  if not find_relations_between(A, B):
    → 叙事间隙：为什么 A 和 B 在同一故事中但没有直接关系？

  用 semantic_search 填补间隙：
    query = f"{A.name} {B.name} {core.name} 的故事"
    → 寻找连接 A 和 B 的合理叙事

  可能发现的关系类型：
  - "enables" — A 使 B 成为可能
  - "motivates" — A 是 B 的动机
  - "happens_during" — A 发生在 B 期间
  - "leads_to" — A 导致了 B
  - "contrasts_with" — A 与 B 形成对比
```

### 4. 多视角叙事
```
对每个辅助视角实体 V:
  traverse_graph(start_entity_id=V, max_depth=2)

  如果 V 的子图与核心叙事有交集：
    → 从 V 的视角重新审视交集部分
    → 可能发现核心视角看不到的关系
```

### 5. 叙事整合
```
将所有发现整合成连贯故事：

  save_dream_episode(
    dream_type="narrative",
    entities_explored=[所有涉及的实体],
    relations_found=N,
    summary="叙事摘要（200字以内的故事梗概）",
    insights="叙事揭示的深层关系"
  )
```

## 评分标准
- 发现完整叙事链中的缺失环节 → 高价值
- 从多视角发现隐藏关系 → 高价值
- 纯粹的叙事重述（无新发现） → 低价值
- 叙事越连贯、越有意外转折 → 越有价值
