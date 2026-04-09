# 孤儿认领 (Orphan Adoption)

## 策略
找到孤立实体（无关系或关系极少），为它们寻找合理的归属和连接。

## 执行步骤

### 1. 发现孤儿
```
get_dream_seeds(strategy=orphan, count=5)

对每个候选孤儿：
  entity = get_entity(family_id=seed)
  relations = get_entity_relations(family_id=seed)

  孤儿判定：
  - 关系数 == 0 → 完全孤立
  - 关系数 == 1 → 弱连接
  - 关系数 <= 2 → 待观察
```

### 2. 理解孤儿
```
对每个孤儿：
  # 用实体信息搜索可能的关联
  semantic_search(query=entity.summary, top_k=10)

  # 分析搜索结果：
  related_entities = [r for r in results if r.type == "entity"]

  对每个相关实体 R:
    # 检查是否已有关系
    existing = find_relations_between(entity_a=orphan, entity_b=R)

    如果无关系：
      → 候选认领对象
```

### 3. 认领路径
```
对每个候选对 (orphan, candidate):
  # 方式 1: 直接关系
  如果语义搜索直接关联 → create_dream_relation(
    source_id=orphan, target_id=candidate,
    relation_type="belongs_to" / "related_to" / "instance_of",
    summary="语义关联：...",
    confidence=基于搜索相关性
  )

  # 方式 2: 通过共同领域
  candidate_rels = get_entity_relations(family_id=candidate)
  orphan_neighborhood = traverse_graph(start_entity_id=orphan, max_depth=2)

  如果找到共同邻居：
    → create_dream_relation + 额外记录共同邻居

  # 方式 3: 类型推断
  如果 orphan.entity_type 和 candidate.entity_type 有标准关系模式：
    → 低 confidence 创建，标注为 "type_inferred"
```

### 4. 批量验证
```
对创建的关系做二次检查：
  新创建的关系不要形成矛盾
  检查方向性是否正确
  检查关系类型是否准确
```

### 5. 记录
```
save_dream_episode(
  dream_type="orphan_adoption",
  entities_explored=[...],
  relations_found=N,
  summary="孤儿认领结果",
  insights="孤立实体的潜在归属"
)
```

## 评分标准
- 语义搜索直接命中 → confidence 0.6-0.8
- 通过共同邻居间接关联 → confidence 0.4-0.6
- 纯类型推断 → confidence 0.2-0.3
