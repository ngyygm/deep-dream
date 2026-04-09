# Hub 重混 (Hub Remix)

## 策略
重新审视图谱中的核心实体（hub），发现其关系网络中被忽视的连接和模式。

## 执行步骤

### 1. 识别 Hub
```
get_dream_seeds(strategy=hub, count=3)

对每个 hub：
  entity = get_entity(family_id=hub)
  relations = get_entity_relations(family_id=hub, limit=50)

  Hub 特征：
  - 关系数 >= 10
  - 连接多个不同类型的实体
  - 在多条最短路径上
```

### 2. 网络拓扑分析
```
对 hub 的邻居做分类：
  neighbors_by_type = group_by(entity_type)
  neighbors_by_relation = group_by(relation_type)

  寻找模式：
  1. 邻居之间是否有缺失的连接？
     for A, B in neighbors:
       if not find_relations_between(A, B):
         → 候选桥接

  2. 关系类型是否过于单一？
     如果 hub 只有 "related_to" 类型 → 需要细化

  3. 是否有"影子邻居"？
     语义相似但未连接的实体对
```

### 3. 关系细化
```
对 hub 的关系做质量评估：

  对每条关系 R:
    if R.relation_type in ["related_to", "connected_to"]:  # 过于模糊
      # 尝试用语义分析推断更精确的关系类型
      semantic_search(query=f"{source.name} 和 {target.name} 的具体关系")
      → 可能细化为 "works_at", "founded", "influences" 等

      create_dream_relation(
        source_id=R.source, target_id=R.target,
        relation_type=更精确的类型,
        summary="从模糊关系细化",
        confidence=0.4,
        dream_type="hub_remix"
      )
```

### 4. 缺失连接发现
```
hub 的邻居 A 和 B 如果没有直接关系：
  path_AB = find_shortest_path(A, B)
  if path_AB and len(path_AB) > 2:
    # 它们通过 hub 才能连接 → 可能应该有直接关系
    semantic_search(query=f"{A.name} {B.name}", top_k=3)
    → 如果有语义支撑，创建直接关系
```

### 5. 记录
```
save_dream_episode(
  dream_type="hub_remix",
  entities_explored=[hub_id, ...neighbors],
  relations_found=N,
  summary="Hub 网络重混结果",
  insights="被忽视的连接模式"
)
```

## 评分标准
- 发现 hub 邻居间的缺失连接 → 高价值
- 细化模糊关系类型 → 中等价值
- 重复已有关系模式 → 低价值
