# 章节写作模板（Fantasyland 记忆增强版）

## 第 X 章：[章节标题]

---

### Step 0: 写前准备

```
# 读取本章大纲
quick_search(query="第X章大纲")

# 读取前文状态
ask(question="上一章结束时各角色的状态、位置、装备")

# 检查伏笔
quick_search(query="未回收的伏笔")

# 获取出场角色画像
batch_profiles(family_ids=[...])
```

### 本章任务

- 本章目标：
- 本章冲突：
- 出场角色：
- 场景地点：
- 本章新信息：
- 本章章末拉力：
- 伏笔操作：□ 埋设 □ 铺垫 □ 回收

---

### Step 1: 场景预设 → remember 写入

```
remember(text="""
第X章场景预设：
- 开场：
- 角色状态：
- 情绪基调：
- 必须推进的伏笔：
- 需要埋下的新伏笔：
""", source_name="预设-第X章", wait=true)
```

---

### Step 2: 段落循环

对每个段落重复以下步骤：

#### 段落 P{n}: [段落简述]

**2a. 记忆召回**

```
quick_search(query="[当前场景相关信息]")
ask(question="[角色状态确认]")
```

**2b. 写作（50-300字）**

> [正文内容]

**2c. 一致性校验**

```
ask(question="校验这段内容是否和已有记忆矛盾：
1. 角色属性是否一致？
2. 角色状态是否衔接？
3. 世界观是否合规？
4. 时间线是否连续？
5. 因果是否合理？
6. 信息获取是否合理？")
```

校验结果：□ 通过 □ 需修改

**2d. 存入记忆**

```
remember(text="[段落正文]", source_name="正文-第X章-P{n}", wait=true)
```

---

### 场景拆分

#### 场景 1

- 主角目标：
- 阻碍：
- 结果变化：
- **记忆检查点：** `quick_search(query="[场景相关]")`

#### 场景 2

- 主角目标：
- 阻碍：
- 结果变化：
- **记忆检查点：** `quick_search(query="[场景相关]")`

#### 场景 3

- 主角目标：
- 阻碍：
- 结果变化：
- **记忆检查点：** `quick_search(query="[场景相关]")`

---

### Step 3: 章节收尾

```
# 存储完整章节
remember(text=chapter_full_text,
         source_name="正文-第X章-[章节标题]",
         wait=true)

# 记录状态变更
remember(text="第X章结束状态变更：...", 
         source_name="状态-第X章结束",
         wait=true)

# 更新角色
evolve_entity_summary(family_id="ent_xxx")

# Dream 探索
dream_run(strategy="narrative", seed_count=3)
```

---

### 写后自查

#### 质量检查

- [ ] 前几段有没有抓手？
- [ ] 本章有没有不可删除的事件？
- [ ] 主角有没有在主动做事？
- [ ] 本章结尾有没有继续读的理由？
- [ ] 对白是否带身份感和信息量？
- [ ] 去了 AI 味？

#### 一致性检查

- [ ] 角色外貌和设定一致？
- [ ] 角色状态和上章衔接？
- [ ] 世界观规则没有违反？
- [ ] 时间线连续？
- [ ] 没有信息泄露（角色知道不该知道的）？
- [ ] 伏笔状态已更新？

#### 记忆写入检查

- [ ] 每段都已 `remember` 存入
- [ ] 完整章节已 `remember` 存入
- [ ] 状态变更已 `remember` 存入
- [ ] 角色摘要已 `evolve_entity_summary` 更新
- [ ] 已 `dream_run` 发现新关联
