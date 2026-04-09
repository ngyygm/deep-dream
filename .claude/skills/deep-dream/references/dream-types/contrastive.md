# 对比发现 (Contrastive)

## 策略
选择相似但不相同的实体对，通过对比发现它们的差异、对立、或演化关系。

## 执行步骤

### 1. 识别相似实体
```
get_dream_seeds(strategy=random, count=5)

对每个种子：
  candidates = find_candidates(description=seed.summary, top_k=3)

  # 过滤：排除自身和已直接关联的实体
  筛选出：
  - 同类型但名字不同的实体
  - summary 中有重叠关键词的实体
```

### 2. 深度对比
```
对每个候选对 (A, B):
  entity_A = get_entity(family_id=A)
  entity_B = get_entity(family_id=B)

  relations_A = get_entity_relations(family_id=A)
  relations_B = get_entity_relations(family_id=B)

  对比维度：
  1. 关系网络差异
     - A 的邻居中哪些 B 不认识？反之亦然
     - 共同邻居有什么特殊含义

  2. 属性差异
     - 类型相同但属性值不同
     - 属性互补的方面

  3. 时间线差异
     - timeline_A vs timeline_B
     - 是否有先后顺序暗示因果关系
```

### 3. 发现关系
```
基于对比，可能发现的关系类型：
  - "contrasts_with" — 直接对立
  - "evolves_into" — A 是 B 的前身
  - "complements" — 互补关系
  - "parallels" — 并行/类比关系
  - "supersedes" — 取代关系

create_dream_relation(
  source_id=A, target_id=B,
  relation_type=上述之一,
  summary="对比发现：...",
  confidence=基于对比深度 0.4-0.7,
  dream_type="contrastive"
)
```

### 4. 记录
```
save_dream_episode(
  dream_type="contrastive",
  entities_explored=[...],
  relations_found=N,
  summary="对比分析",
  insights="相似实体间的关键差异"
)
```

## 评分标准
- 同类型实体的细微差异 → 高价值
- 跨类型但有语义重叠 → 中等价值
- 纯属性差异无深层含义 → 低价值
