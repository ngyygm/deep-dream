# 自由联想 (Free Association)

## 策略
从种子实体出发，沿关系链自由漫步，每一步选择最有"联想感"的邻居继续探索。

## 执行步骤

### 1. 获取种子
```
get_dream_seeds(strategy=random, count=3)
```

### 2. 对每个种子
```
seed = get_entity(family_id)
print(f"从 {seed.name} 出发")
```

### 3. 漫步（最多 5 步）
每一步：
```
neighbors = traverse_graph(start_entity_id, direction=both, max_depth=1)

选择标准（按优先级）：
1. 关系类型最模糊/抽象的邻居（如 "influences", "related_to"）
2. 跨类型的邻居（Person → Concept, Organization → Event）
3. 关系最少的方向（冷门路径优先）
4. 避免回溯已访问实体

对当前实体和所选邻居：
  - semantic_search(query="两者可能的联系", top_k=3)
  - 如果搜索结果中有支撑证据：
    - create_dream_relation(
        source_id=current,
        target_id=neighbor,
        relation_type="free_association",
        summary="发现理由...",
        confidence=基于证据强度 0.3-0.8,
        dream_type="free_association"
      )
```

### 4. 记录
```
save_dream_episode(
  dream_type="free_association",
  entities_explored=[...],
  relations_found=N,
  summary="漫步路径和发现",
  insights="关键洞察"
)
```

## 评分标准
- 路径上每一步有语义支撑 → +0.1 confidence
- 跨越 2+ 个实体类型 → +0.1
- 发现 unexpected 连接 → +0.2
