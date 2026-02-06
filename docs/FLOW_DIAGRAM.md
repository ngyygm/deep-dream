# TMG 系统流程图

> 完整的处理流程，从读取文件到处理结束

## 📊 整体流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              整体处理流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │  输入文件    │
                              │  (小说/文档) │
                              └──────┬──────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  1. 滑动窗口处理器     │
                         │  SlidingWindowProcessor│
                         └───────────┬───────────┘
                                     │
                                     │ 生成多个 WindowChunk
                                     ▼
                    ┌────────────────────────────────────┐
                    │                                    │
                    │    循环处理每个 WindowChunk        │
                    │                                    │
                    │  ┌──────────────────────────────┐  │
                    │  │                              │  │
                    │  │    2. Memory Agent          │  │
                    │  │    处理单个窗口              │  │
                    │  │                              │  │
                    │  └──────────────────────────────┘  │
                    │                                    │
                    └────────────────┬───────────────────┘
                                     │
                                     │ 每个窗口产生 Commit
                                     ▼
                         ┌───────────────────────┐
                         │  3. Git-Style Graph   │
                         │  图谱版本管理          │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  4. Storage Layer     │
                         │  持久化存储           │
                         └───────────────────────┘
```

---

## 🔄 详细流程图

### Phase 1: 文件输入与窗口切分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: 原始文件                                                             │
│  - 文件路径: str                                                             │
│  - 文件类型: novel/document/chat...                                         │
│  - 元数据: {title, author, source...}                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Module: SlidingWindowProcessor                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  配置:                                                                       │
│  - window_size: 512 字符                                                     │
│  - overlap_before: 100 字符                                                  │
│  - overlap_after: 50 字符                                                    │
│  - sentence_aware: true                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  处理逻辑:                                                                   │
│  1. 读取文件内容                                                             │
│  2. 按窗口大小切分，保留重叠区域                                              │
│  3. 避免在句子中间截断                                                       │
│  4. 为每个窗口分配 sequence_index                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: List[WindowChunk]                                                   │
│                                                                              │
│  WindowChunk:                                                                │
│  - content: str           # 窗口内容                                         │
│  - start_pos: int         # 起始位置                                         │
│  - end_pos: int           # 结束位置                                         │
│  - overlap_before: str    # 前向重叠                                         │
│  - overlap_after: str     # 后向重叠                                         │
│  - sequence_index: int    # 序列索引                                         │
│  - world_time: datetime   # 处理时间                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 2: Memory Agent 处理单个窗口

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT:                                                                      │
│  - chunk: WindowChunk                    # 当前窗口                          │
│  - memory_cache: MemoryCache             # 当前记忆缓存                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 实体抽取与对齐（基于 MemoryCache）                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - chunk.content: str                    # 当前文本                          │
│  - memory_cache.content: str             # 完整的 cache 文档                 │
│  - memory_cache.active_entities          # 活跃实体列表                      │
│  - memory_cache.pronoun_context          # 代词上下文                        │
│                                                                              │
│  调用: LLM.extract_and_align_entities()                                     │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. LLM 从文本中识别所有实体                                                 │
│  2. 与 MemoryCache 中的活跃实体进行初步对齐                                  │
│  3. 解析代词指向                                                             │
│                                                                              │
│  OUTPUT: List[EntityResult]                                                  │
│  - mention: str            # 文本中的表述                                    │
│  - name: str               # 实体名称                                        │
│  - description: str        # 完整描述                                        │
│  - matched_existing: bool  # 是否匹配到 MemoryCache 中的实体                 │
│  - matched_entity_id: str  # 匹配的实体ID（如果匹配）                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 关系抽取（基于已抽取的实体）                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - chunk.content: str                    # 当前文本                          │
│  - entity_results: List[EntityResult]    # 已抽取的实体                      │
│                                                                              │
│  调用: LLM.extract_relations()                                              │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. LLM 基于已抽取的实体，从文本中抽取关系                                   │
│  2. 生成自然语言描述的概念边                                                 │
│  3. 提取事件时间线索                                                         │
│                                                                              │
│  OUTPUT: List[RelationResult]                                                │
│  - source_entity: EntityResult   # 源实体                                    │
│  - target_entity: EntityResult   # 目标实体                                  │
│  - description: str              # 概念边描述                                │
│  - event_time_hint: str          # 事件时间线索                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 图谱搜索（查找已存在的实体/关系）                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - entity_results: List[EntityResult]      # 已抽取的实体                    │
│  - relation_results: List[RelationResult]  # 已抽取的关系                    │
│                                                                              │
│  调用:                                                                       │
│  - GraphManager.search_entities()          # 搜索实体                        │
│  - GraphManager.get_entity_history()       # 获取实体历史版本                │
│  - GraphManager.get_edge_history_between() # 获取关系历史版本                │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. 对每个抽取的实体，在图谱中搜索是否已存在                                  │
│  2. 对每个抽取的关系，在图谱中搜索是否已存在                                  │
│  3. 获取已有实体/关系的历史版本                                              │
│                                                                              │
│  OUTPUT: RelatedKnowledge                                                    │
│  - entities: List[Entity]        # 图谱中匹配的实体（含历史版本）            │
│  - edges: List[ConceptEdge]      # 图谱中匹配的关系（含历史版本）            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: 判断更新（基于图谱搜索结果）                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - entity_results: List[EntityResult]       # 已抽取的实体                   │
│  - relation_results: List[RelationResult]   # 已抽取的关系                   │
│  - related_knowledge: RelatedKnowledge      # 图谱搜索结果                   │
│                                                                              │
│  调用: LLM.judge_entity_update() / LLM.judge_edge_update()                  │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. 比较新抽取的实体/关系与图谱中已有的                                      │
│  2. LLM 判断：NEW / UPDATE / REDUNDANT / CONFLICT                           │
│                                                                              │
│  OUTPUT: UpdateDecisions                                                     │
│  - entity_decisions: List[UpdateDecision]   # 实体更新决策                   │
│  - edge_decisions: List[UpdateDecision]     # 关系更新决策                   │
│                                                                              │
│  UpdateDecision:                                                             │
│  - type: NEW/UPDATE/REDUNDANT/CONFLICT                                       │
│  - target: EntityResult/RelationResult                                       │
│  - reasoning: str                                                            │
│  - target_version_id: str (如果是UPDATE)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: 推断事件时间（基于 MemoryCache）                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - chunk.content: str                    # 当前文本                          │
│  - memory_cache.content: str             # 完整的 cache 文档                 │
│                                                                              │
│  调用: LLM.infer_event_time()                                               │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. 提取文本中明确的时间标记                                                 │
│  2. 基于 cache 中的章节/场景信息推断相对时间                                 │
│  3. 生成 EventTime 对象                                                      │
│                                                                              │
│  OUTPUT: Dict[str, EventTime]                                                │
│  - key: entity_id 或 edge_id                                                 │
│  - value: EventTime (anchor_type, anchor_value, sequence_index...)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 6: 更新记忆缓存（文档化更新）                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - chunk.content: str                    # 当前文本                          │
│  - memory_cache.content: str             # 当前 cache 内容                   │
│  - entity_results: List[EntityResult]    # 新抽取的实体                      │
│  - relation_results: List[RelationResult]# 新抽取的关系                      │
│  - update_decisions: UpdateDecisions     # 更新决策                          │
│  - cache_update_rules: str               # cache 更新规则                    │
│                                                                              │
│  调用:                                                                       │
│  - LLM.update_memory_cache_content()     # 生成更新后的 cache 内容           │
│  - MemoryCacheManager.check_if_changed() # 检查是否有变化                    │
│  - MemoryCacheManager.save_cache()       # 保存新版本（如果有变化）          │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. LLM 基于规则生成更新后的 cache 文档                                      │
│  2. 检查内容是否有变化                                                       │
│  3. 如果有变化，保存新版本的 cache                                           │
│                                                                              │
│  OUTPUT:                                                                     │
│  - updated_cache: MemoryCache            # 更新后的 cache                    │
│  - cache_changed: bool                   # 是否有变化                        │
│  - cache_id: str                         # cache 版本ID（新生成或保持不变）  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 7: 执行 Commit（只有有变化才提交）                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│  - entity_results: List[EntityResult]    # 实体抽取结果                      │
│  - relation_results: List[RelationResult]# 关系抽取结果                      │
│  - update_decisions: UpdateDecisions     # 更新决策                          │
│  - event_times: Dict[str, EventTime]     # 事件时间                          │
│  - chunk: WindowChunk                    # 当前窗口                          │
│  - memory_cache_id: str                  # 使用的 cache 版本ID               │
│                                                                              │
│  调用:                                                                       │
│  - GraphManager.commit()                 # 执行提交                          │
│  - VersionManager.create_entity_version()# 创建实体版本                      │
│  - VersionManager.create_edge_version()  # 创建边版本                        │
│                                                                              │
│  处理逻辑:                                                                   │
│  1. 过滤掉 REDUNDANT 的决策                                                  │
│  2. 如果没有 NEW 或 UPDATE，返回 None（不提交）                              │
│  3. 为 NEW 决策创建新实体/边                                                 │
│  4. 为 UPDATE 决策创建新版本                                                 │
│  5. 创建 Commit 记录，引用 memory_cache_id                                   │
│                                                                              │
│  OUTPUT: Optional[Commit]                                                    │
│  - id: str                               # commit ID                         │
│  - world_time: datetime                  # 提交时间                          │
│  - added_entity_versions: List[str]      # 新增实体版本                      │
│  - modified_entity_versions: List[str]   # 修改的实体版本                    │
│  - added_edge_versions: List[str]        # 新增边版本                        │
│  - modified_edge_versions: List[str]     # 修改的边版本                      │
│  - memory_cache_id: str                  # 引用的 cache 版本                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: CommitResult                                                        │
│  - commit: Optional[Commit]   # 提交结果（可能为 None）                      │
│  - cache: MemoryCache         # 更新后的 cache                               │
│  - cache_changed: bool        # cache 是否有变化                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 3: 循环处理所有窗口

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  主循环                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

初始化:
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. 加载或创建初始 MemoryCache                                               │
│  2. 初始化 MemoryAgent                                                       │
│  3. 初始化 GitStyleGraphManager                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FOR each chunk in chunks:                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  result = await memory_agent.process_chunk(chunk)                     │  │
│  │                                                                       │  │
│  │  // 输出:                                                             │  │
│  │  // - result.commit: 本次提交（可能为 None）                          │  │
│  │  // - result.cache: 更新后的 cache                                    │  │
│  │  // - result.cache_changed: cache 是否有变化                          │  │
│  │                                                                       │  │
│  │  // 更新进度                                                          │  │
│  │  yield ProcessingProgress(                                            │  │
│  │      current_chunk=chunk.sequence_index,                              │  │
│  │      total_chunks=len(chunks),                                        │  │
│  │      entities_found=...,                                              │  │
│  │      relations_found=...,                                             │  │
│  │      commit_id=result.commit.id if result.commit else None            │  │
│  │  )                                                                    │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  END FOR                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  处理完成                                                                    │
│                                                                              │
│  输出统计:                                                                   │
│  - 总实体数                                                                  │
│  - 总关系数                                                                  │
│  - 总提交数                                                                  │
│  - 最终 cache 状态                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            模块依赖图                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              TMG (主入口)                                    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SlidingWindowProcessor                            │    │
│  │                    (无外部依赖)                                       │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         MemoryAgent                                  │    │
│  │                             │                                        │    │
│  │      ┌──────────────────────┼──────────────────────┐                │    │
│  │      │                      │                      │                │    │
│  │      ▼                      ▼                      ▼                │    │
│  │  ┌────────┐           ┌──────────┐          ┌───────────┐          │    │
│  │  │  LLM   │           │  Graph   │          │   Cache   │          │    │
│  │  │ Client │           │ Manager  │          │  Manager  │          │    │
│  │  └────────┘           └────┬─────┘          └─────┬─────┘          │    │
│  │                            │                      │                │    │
│  └────────────────────────────┼──────────────────────┼────────────────┘    │
│                               │                      │                      │
│                               ▼                      ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Storage Layer                                 │    │
│  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │    │
│  │   │    SQLite     │  │  Cache Files  │  │  Vector DB    │          │    │
│  │   │ (entities,    │  │  (.md files)  │  │  (optional)   │          │    │
│  │   │  edges,       │  │               │  │               │          │    │
│  │   │  commits)     │  │               │  │               │          │    │
│  │   └───────────────┘  └───────────────┘  └───────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 各模块接口定义

### 1. TMG (主入口)

```python
class TemporalMemoryGraph:
    """主入口类"""
    
    async def process_text(
        self,
        text: str,
        source_type: str,
        metadata: Dict
    ) -> AsyncIterator[ProcessingProgress]:
        """
        处理文本（流式输出进度）
        
        INPUT:
        - text: str                 # 原始文本
        - source_type: str          # 类型 (novel, chat, news...)
        - metadata: Dict            # 元数据 {title, author...}
        
        OUTPUT (yield):
        - ProcessingProgress        # 进度信息
        """
        pass
    
    async def query(
        self,
        question: str,
        temporal_constraint: Optional[TemporalConstraint] = None
    ) -> QueryResult:
        """
        查询知识图谱
        
        INPUT:
        - question: str             # 自然语言问题
        - temporal_constraint       # 时间约束（可选）
        
        OUTPUT:
        - QueryResult               # 查询结果
        """
        pass
```

### 2. SlidingWindowProcessor

```python
class SlidingWindowProcessor:
    """滑动窗口处理器"""
    
    def process_text(self, text: str) -> Iterator[WindowChunk]:
        """
        切分文本为窗口
        
        INPUT:
        - text: str                 # 原始文本
        
        OUTPUT:
        - Iterator[WindowChunk]     # 窗口迭代器
        """
        pass
```

### 3. MemoryAgent

```python
class MemoryAgent:
    """记忆代理"""
    
    async def process_chunk(self, chunk: WindowChunk) -> CommitResult:
        """
        处理单个窗口
        
        INPUT:
        - chunk: WindowChunk        # 文本窗口
        
        OUTPUT:
        - CommitResult              # 处理结果
          - commit: Optional[Commit]
          - cache: MemoryCache
          - cache_changed: bool
        
        调用的模块:
        - LLM Client (实体抽取、关系抽取、更新判断、时间推断、cache更新)
        - GraphManager (搜索、commit)
        - CacheManager (保存cache)
        """
        pass
```

### 4. MemoryCacheManager

```python
class MemoryCacheManager:
    """记忆缓存管理器"""
    
    def save_cache(self, cache: MemoryCache) -> str:
        """
        保存 cache
        
        INPUT:
        - cache: MemoryCache        # 缓存对象
        
        OUTPUT:
        - str                       # cache_id (cache_{timestamp}_{hash})
        """
        pass
    
    def load_cache(self, cache_id: str) -> MemoryCache:
        """
        加载 cache
        
        INPUT:
        - cache_id: str             # 缓存ID
        
        OUTPUT:
        - MemoryCache               # 缓存对象
        """
        pass
    
    def check_if_changed(self, old_cache: MemoryCache, new_content: str) -> bool:
        """
        检查是否有变化
        
        INPUT:
        - old_cache: MemoryCache    # 原缓存
        - new_content: str          # 新内容
        
        OUTPUT:
        - bool                      # 是否有变化
        """
        pass
```

### 5. GitStyleGraphManager

```python
class GitStyleGraphManager:
    """图谱管理器"""
    
    def commit(
        self,
        entity_changes: List[EntityChange],
        edge_changes: List[EdgeChange],
        event_times: Dict[str, EventTime],
        source_type: str,
        source_text_range: Tuple[int, int],
        source_text_snippet: str,
        memory_cache_id: str,
        message: str = ""
    ) -> Optional[Commit]:
        """
        执行提交
        
        INPUT:
        - entity_changes            # 实体变更列表
        - edge_changes              # 边变更列表
        - event_times               # 事件时间
        - source_type               # 来源类型
        - source_text_range         # 源文本范围
        - source_text_snippet       # 源文本片段
        - memory_cache_id           # 引用的cache版本
        - message                   # 提交说明
        
        OUTPUT:
        - Optional[Commit]          # 提交对象（无变化时为None）
        
        调用的模块:
        - VersionManager (创建实体/边版本)
        - Storage (持久化)
        """
        pass
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """搜索实体"""
        pass
    
    def get_entity_history(self, entity_id: str) -> List[EntityVersion]:
        """获取实体历史版本"""
        pass
    
    def get_edge_history_between(
        self, 
        source_id: str, 
        target_id: str
    ) -> List[EdgeVersion]:
        """获取两个实体之间关系的历史"""
        pass
```

### 6. LLM Client

```python
class LLMClient:
    """LLM 客户端"""
    
    async def extract_and_align_entities(
        self,
        text: str,
        memory_cache_content: str,
        active_entities: List[Dict],
        pronoun_context: Dict[str, str],
        current_context: str
    ) -> List[EntityResult]:
        """实体抽取与对齐"""
        pass
    
    async def extract_relations(
        self,
        text: str,
        entities: List[EntityResult]
    ) -> List[RelationResult]:
        """关系抽取"""
        pass
    
    async def judge_entity_update(
        self,
        new_info: EntityResult,
        existing_entity: Entity,
        existing_versions: List[EntityVersion]
    ) -> UpdateDecision:
        """判断实体更新"""
        pass
    
    async def judge_edge_update(
        self,
        new_info: RelationResult,
        existing_edge: ConceptEdge,
        existing_versions: List[EdgeVersion]
    ) -> UpdateDecision:
        """判断关系更新"""
        pass
    
    async def infer_event_time(
        self,
        text: str,
        memory_cache_content: str
    ) -> EventTime:
        """推断事件时间"""
        pass
    
    async def update_memory_cache_content(
        self,
        current_cache_content: str,
        new_text: str,
        new_entities: List[EntityResult],
        new_relations: List[RelationResult],
        update_decisions: UpdateDecisions,
        cache_update_rules: str
    ) -> str:
        """更新记忆缓存内容"""
        pass
```

---

## 📈 数据流向图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据流向                                        │
└─────────────────────────────────────────────────────────────────────────────┘

原始文本
    │
    ▼
┌─────────────┐     ┌─────────────┐
│   Window    │────▶│   Entity    │
│   Chunks    │     │   Results   │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐     ┌──────────────┐
                    │  Relation   │────▶│   Related    │
                    │   Results   │     │  Knowledge   │
                    └─────────────┘     │ (from Graph) │
                                        └──────┬───────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Update    │
                                        │  Decisions  │
                                        └──────┬──────┘
                                               │
              ┌────────────────────────────────┼───────────────────────────┐
              │                                │                           │
              ▼                                ▼                           ▼
       ┌─────────────┐                  ┌─────────────┐            ┌──────────────┐
       │   Event     │                  │   Updated   │            │    Commit    │
       │   Times     │                  │   Cache     │            │   (Graph)    │
       └─────────────┘                  └─────────────┘            └──────────────┘
              │                                │                           │
              │                                │                           │
              └────────────────────────────────┼───────────────────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Storage   │
                                        │  (SQLite +  │
                                        │   Files)    │
                                        └─────────────┘
```

---

## 🔄 状态机

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           处理状态机                                         │
└─────────────────────────────────────────────────────────────────────────────┘

          ┌─────────────┐
          │    IDLE     │
          └──────┬──────┘
                 │ process_text()
                 ▼
          ┌─────────────┐
          │  PREPARING  │ ─────── 初始化 cache, agent, graph
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
     ┌───▶│ PROCESSING  │ ─────── 处理当前窗口
     │    │   CHUNK     │
     │    └──────┬──────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │  EXTRACTING │ ─────── 实体+关系抽取
     │    └──────┬──────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │  SEARCHING  │ ─────── 搜索图谱
     │    └──────┬──────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │   JUDGING   │ ─────── 判断更新
     │    └──────┬──────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │  UPDATING   │ ─────── 更新 cache
     │    │   CACHE     │
     │    └──────┬──────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │ COMMITTING  │ ─────── 执行 commit
     │    └──────┬──────┘
     │           │
     │           │ 还有下一个窗口?
     │    YES    │    NO
     └───────────┤
                 │
                 ▼
          ┌─────────────┐
          │  COMPLETED  │
          └─────────────┘
```

---

## 📝 使用示例

```python
import asyncio
from tmg import TemporalMemoryGraph

async def main():
    # 1. 初始化
    tmg = TemporalMemoryGraph(
        storage_path="./my_knowledge.db",
        llm_config={"provider": "openai", "model": "gpt-4"}
    )
    
    # 2. 读取文件
    with open("novel.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # 3. 处理文本（流式）
    async for progress in tmg.process_text(
        text=text,
        source_type="novel",
        metadata={"title": "我的小说", "author": "作者名"}
    ):
        print(f"进度: {progress.current_chunk}/{progress.total_chunks}")
        print(f"实体数: {progress.entities_found}")
        print(f"关系数: {progress.relations_found}")
        if progress.commit_id:
            print(f"Commit: {progress.commit_id}")
    
    # 4. 处理完成
    print(f"处理完成！")
    print(f"总实体数: {tmg.entity_count}")
    print(f"总关系数: {tmg.edge_count}")
    print(f"总提交数: {tmg.commit_count}")

asyncio.run(main())
```
