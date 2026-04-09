# 跳跃发现 (Leap)

## 策略
故意忽略中间节点，尝试在距离遥远的实体间直接建立关系。适合发现"范式转移"级别的连接。

## 执行步骤

### 1. 获取种子
```
get_dream_seeds(strategy=hub, count=2)  ← 选 hub 实体
get_dream_seeds(strategy=orphan, count=2) ← 选孤立实体

混合：1 hub + 1 orphan 为一组
```

### 2. 远距离配对
```
对每组 (hub, orphan):
  path = find_shortest_path(from_entity=hub, to_entity=orphan, max_depth=6)

  如果 depth >= 3:
    → 跳跃距离足够，有发现潜力

  如果无路径:
    → 尝试语义桥接
```

### 3. 语义桥接
```
对远距离对 (A, B):
  # 用两者的 summary 拼接搜索
  search_A = get_entity(family_id=A).summary
  search_B = get_entity(family_id=B).summary

  semantic_search(query=f"{search_A} {search_B}", top_k=5)

  # 分析搜索结果：
  - 如果返回的相关实体同时连接到 A 和 B → 强信号
  - 如果返回的实体只与一方相关 → 弱信号
  - 如果无相关结果 → 可能不是有效连接
```

### 4. 创造性推理
```
基于语义搜索结果，尝试推理 A→B 的可能关系：

  推理模板：
  - A [影响了/启发了] B，因为 ...
  - A 和 B [共享/对立] 于 ..., 因为 ...
  - 在 ... 语境下，A 可以被视为 B 的 ...

  每个推理必须附上 evidence（搜索结果或路径）
```

### 5. 记录
```
save_dream_episode(
  dream_type="leap",
  entities_explored=[...],
  relations_found=N,
  summary="跳跃发现",
  insights="远距离连接的推理过程"
)
```

## 评分标准
- 跳跃距离 (depth) 越远，发现越有价值
- 但 confidence 应该与距离成反比（远距离 = 低 confidence）
- depth=3: confidence 上限 0.7
- depth=4+: confidence 上限 0.5
- 无路径纯语义推理: confidence 上限 0.3
