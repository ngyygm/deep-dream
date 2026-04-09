# 时间桥接 (Temporal Bridge)

## 策略
沿实体的版本历史（timeline）追溯，发现跨时间片的因果、演化、或周期性关系。

## 执行步骤

### 1. 选择目标
```
get_dream_seeds(strategy=recent, count=3)

对每个种子：
  timeline = get_entity_timeline(family_id=seed, limit=20)

  选择条件：
  - timeline 长度 >= 3（有足够历史）
  - 最近有变化（活跃实体优先）
```

### 2. 时间线分析
```
对每个目标的 timeline:
  versions = get_entity_versions(family_id=seed, limit=10)

  分析：
  1. 版本间的关键变化
     - diff = get_entity_version_diff(from_version=v1, to_version=v2)
  2. 变化发生的时间点
  3. 变化触发源（source 字段）
```

### 3. 因果推理
```
寻找时间上的前后关系：

对实体 A (时间线 tA) 和实体 B (时间线 tB):
  如果 A 在 tA[k] 的变化 紧接着 B 在 tB[m] 出现变化：
    → 候选因果关系

  验证：
  1. semantic_search(query="A 的变化对 B 的影响")
  2. find_relations_between(entity_a=A, entity_b=B)
  3. 如果已有关系，检查关系是否反映了因果

  可能发现：
  - "causes" — A 导致了 B 的变化
  - "precedes" — A 先于 B 发生
  - "triggers" — A 触发了 B 的创建/更新
  - "responds_to" — B 是对 A 变化的响应
```

### 4. 周期性检测
```
对有长历史的实体：
  检查 summary 在不同版本间的循环变化
  如果发现周期性 → 记录为 "periodic_pattern" 关系
```

### 5. 记录
```
save_dream_episode(
  dream_type="temporal_bridge",
  entities_explored=[...],
  relations_found=N,
  summary="时间线分析发现",
  insights="因果或演化关系"
)
```

## 评分标准
- 明确的时间先后 + 语义因果 → confidence 0.6-0.8
- 时间接近但因果模糊 → confidence 0.3-0.5
- 仅基于时间相邻 → confidence 0.1-0.3
