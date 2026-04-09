# 跨领域桥接 (Cross Domain)

## 策略
选择来自不同领域/类型的实体对，寻找它们之间隐性但合理的跨领域关系。

## 执行步骤

### 1. 识别领域
```
get_stats → 了解实体类型分布
get_dream_seeds(strategy=random, count=5)

按 entity_type 分组种子：
  groups = {type: [entities] for type, entity in seeds}
  至少需要 2 个不同类型
```

### 2. 配对
```
从不同类型组中各选 1 个实体，组成候选对

配对优先级：
1. 类型跨度最大（Person + Concept > Person + Organization）
2. 两者关系数差距大（hub + isolated）
3. 从未有过共同邻居的实体对
```

### 3. 探索连接
```
对每个候选对 (A, B):
  path = find_shortest_path(from_entity=A, to_entity=B, max_depth=4)

  如果有路径 (depth >= 2):
    - 分析路径上中间实体的角色
    - semantic_search(query=f"{A.name} 和 {B.name} 的关系")

  如果无路径:
    - semantic_search(query=f"{A.name} {B.name}", top_k=5)
    - 用搜索结果判断是否存在潜在关系

  判断是否值得创建关系：
  - 路径存在且中间节点有意义 → confidence 0.6-0.8
  - 无路径但语义搜索有强支撑 → confidence 0.4-0.6
  - 仅基于类型推断 → confidence 0.2-0.3
```

### 4. 记录
```
save_dream_episode(
  dream_type="cross_domain",
  entities_explored=[...],
  relations_found=N,
  summary="跨领域发现",
  insights="领域间的隐性联系"
)
```

## 评分标准
- 两个实体无直接路径但有语义关联 → 最有价值的发现
- 发现连接两个领域群的桥梁实体 → 高价值
- 纯粹基于类型的泛泛推断 → 低价值
