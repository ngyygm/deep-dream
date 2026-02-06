"""
Orchestrator 编排器

记忆检索 Agent 的主入口，协调 Planner、Executor、Evaluator、Reasoner、Summarizer 的工作
"""
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys

# 添加父目录到路径以导入 processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models import (
    AgentConfig, QueryResult, ToolCall, ToolResult,
    RetrievedMemory, Message
)
from .llm import create_llm_client, BaseLLMClient
from .llm.openai_client import MockLLMClient
from .planner import Planner
from .executor import Executor
from .executor.tool_registry import create_default_registry, ToolRegistry
from .evaluator import Evaluator
from .context import ContextManager, SmartCache, ReasoningCache, QuestionType
from .reasoner import Reasoner
from .summarizer import Summarizer
from .logger import AgentLogger, set_logger


class MemoryRetrievalAgent:
    """
    记忆检索 Agent
    
    使用 ReAct 循环（规划-执行-观察-判断）从时序记忆图谱中检索相关记忆。
    
    新增功能：
    - ReasoningCache: 追踪推理状态，包括子目标、已知事实、缺失信息
    - Reasoner: 分析问题类型，进行推理规划和结论生成
    - Summarizer: 筛选有用信息，生成推理总结
    """
    
    def __init__(
        self,
        storage_paths: Union[str, List[str]] = None,
        storage_managers: List[Any] = None,
        llm_config: Dict[str, Any] = None,
        config: AgentConfig = None,
        verbose: bool = True,
        log_level: str = "moderate"
    ):
        """
        初始化记忆检索 Agent
        
        Args:
            storage_paths: 记忆库路径（字符串或列表）
            storage_managers: StorageManager 实例列表（可选，与 storage_paths 二选一）
            llm_config: LLM 配置字典，包含 api_key, base_url, model 等
            config: AgentConfig 配置对象（可选）
            verbose: 是否打印决策链路
            log_level: 日志级别 (minimal, moderate, verbose)
        """
        # 初始化配置
        self.config = config or AgentConfig()
        if llm_config:
            self.config.llm_api_key = llm_config.get("api_key", "")
            self.config.llm_base_url = llm_config.get("base_url", "https://api.openai.com/v1")
            self.config.llm_model = llm_config.get("model", "gpt-4")
            self.config.llm_temperature = llm_config.get("temperature", 0.7)
            self.config.llm_max_tokens = llm_config.get("max_tokens", 4096)
        
        self.config.verbose = verbose
        self.config.log_level = log_level
        
        # 初始化日志
        self.logger = AgentLogger(
            level=log_level if verbose else "minimal",
            enable_colors=True
        )
        set_logger(self.logger)
        
        # 初始化存储管理器
        self.storage_managers = []
        if storage_managers:
            self.storage_managers = storage_managers
        elif storage_paths:
            paths = [storage_paths] if isinstance(storage_paths, str) else storage_paths
            self._init_storage_managers(paths)
        
        # 初始化 LLM 客户端
        self.llm_client = self._create_llm_client()
        
        # 初始化工具注册表（为每个存储管理器创建）
        self.tool_registries: List[ToolRegistry] = []
        for sm in self.storage_managers:
            self.tool_registries.append(create_default_registry(sm))
        
        # 如果没有存储管理器，创建一个空的注册表用于测试
        if not self.tool_registries:
            self.tool_registries.append(ToolRegistry())
        
        # 初始化推理缓存
        self.reasoning_cache = ReasoningCache()
        
        # 初始化组件
        self.planner = Planner(
            llm_client=self.llm_client,
            tools=self.tool_registries[0].get_all_definitions(),
            logger=self.logger
        )
        
        self.executors = [
            Executor(
                tool_registry=registry,
                parallel=self.config.parallel_tools,
                timeout=self.config.tool_timeout,
                logger=self.logger
            )
            for registry in self.tool_registries
        ]
        
        self.evaluator = Evaluator(
            llm_client=self.llm_client,
            logger=self.logger
        )
        
        # 初始化推理器
        self.reasoner = Reasoner(
            llm_client=self.llm_client,
            reasoning_cache=self.reasoning_cache,
            logger=self.logger
        )
        
        # 初始化总结器
        self.summarizer = Summarizer(
            llm_client=self.llm_client,
            logger=self.logger
        )
        
        # 初始化上下文和缓存
        self.context_manager = ContextManager(llm_client=self.llm_client)
        self.cache = SmartCache() if self.config.enable_cache else None
    
    def _init_storage_managers(self, paths: List[str]):
        """初始化存储管理器"""
        try:
            from processor.storage import StorageManager
            from processor.embedding_client import EmbeddingClient
            
            # 创建 embedding 客户端（如果配置了）
            embedding_client = None
            if self.config.embedding_model_path:
                embedding_client = EmbeddingClient(
                    model_path=self.config.embedding_model_path,
                    device=self.config.embedding_device
                )
            
            for path in paths:
                sm = StorageManager(
                    storage_path=path,
                    embedding_client=embedding_client
                )
                self.storage_managers.append(sm)
                
        except ImportError as e:
            self.logger.warning(f"无法导入 StorageManager: {e}")
            self.logger.warning("请确保 processor 模块可用，或直接传入 storage_managers 参数")
    
    def _create_llm_client(self) -> BaseLLMClient:
        """创建 LLM 客户端"""
        if not self.config.llm_api_key:
            # 没有配置 API Key，使用模拟客户端
            self.logger.warning("未配置 LLM API Key，使用模拟客户端")
            return MockLLMClient()
        
        return create_llm_client(
            provider="custom",
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
    
    def query(
        self,
        messages: Union[List[Dict[str, str]], str],
        enable_reasoning: bool = True,
        **kwargs
    ) -> QueryResult:
        """
        同步查询接口
        
        Args:
            messages: OpenAI 格式的消息列表，或直接传入问题字符串
            enable_reasoning: 是否启用推理功能（分析问题、生成总结）
            **kwargs: 其他参数
            
        Returns:
            QueryResult 对象
        """
        # 标准化输入
        if isinstance(messages, str):
            question = messages
            conversation_history = []
        else:
            # 提取问题（最后一条用户消息）
            question = ""
            conversation_history = []
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                conversation_history.append(msg)
        
        if not question:
            return QueryResult(
                reasoning_trace=[{"error": "未提供问题"}]
            )
        
        # 开始查询
        start_time = time.time()
        self.logger.start_query(question)
        
        # 初始化上下文
        context = self.context_manager.start_query(question)
        
        try:
            # 1. 分析问题（如果启用推理）
            if enable_reasoning:
                self.logger.info("分析问题类型...")
                reasoning_state = self.reasoner.analyze_question(question)
                self.logger.info(f"问题类型: {reasoning_state.question_type.value}")
            else:
                reasoning_state = None
            
            # 2. ReAct 循环
            iteration = 0
            while iteration < self.config.max_iterations:
                iteration += 1
                self.context_manager.increment_iteration()
                self.logger.iteration(iteration, self.config.max_iterations)
                
                # 获取推理状态摘要
                reasoning_state_str = None
                if reasoning_state:
                    reasoning_state_str = self.reasoning_cache.get_state_summary()
                
                # 2.1 规划
                plan = self.planner.plan(
                    question=question,
                    collected_info=self.context_manager.get_collected_info(),
                    reasoning_state=reasoning_state_str
                )
                
                self.context_manager.add_reasoning_step(
                    "plan",
                    plan.get("analysis", ""),
                    {"tool_calls": [tc.tool_name for tc in plan.get("tool_calls", [])]}
                )
                
                # 记录规划
                tool_calls = plan.get("tool_calls", [])
                self.logger.plan(
                    plan.get("analysis", "规划中..."),
                    tool_calls
                )
                
                # 检查是否规划器认为已完成
                if plan.get("is_complete", False) or not tool_calls:
                    self.logger.info("规划器判断信息已充足")
                    break
                
                # 2.2 执行（对所有存储管理器）
                all_results = []
                for executor in self.executors:
                    results = executor.execute(tool_calls)
                    all_results.extend(results)
                    
                    # 添加结果到上下文
                    for result in results:
                        self.context_manager.add_tool_result(result.tool_name, result)
                        
                        # 记录到推理缓存
                        if reasoning_state and result.data:
                            self.reasoning_cache.record_query(
                                tool_name=result.tool_name,
                                parameters={},  # 简化
                                iteration=iteration,
                                success=result.is_success,
                                result_summary=result.data.get("message", "") if isinstance(result.data, dict) else ""
                            )
                
                # 2.3 整合事实（如果启用推理）
                if reasoning_state:
                    self.reasoner.integrate_facts(self.context_manager.get_collected_info())
                
                # 2.4 尝试得出结论
                if reasoning_state:
                    can_conclude, conclusion, confidence = self.reasoner.try_conclude()
                    if can_conclude:
                        self.logger.info(f"推理器得出结论（置信度: {confidence:.0%}）")
                        self.logger.info(f"  [结论] {conclusion[:200]}..." if len(conclusion) > 200 else f"  [结论] {conclusion}")
                        # 输出推理依据
                        state = self.reasoner.cache.state
                        if state and state.known_facts:
                            # 输出推理链
                            reasoning_steps = [(k, v) for k, v in sorted(state.known_facts.items()) if k.startswith("reasoning_step")]
                            if reasoning_steps:
                                self.logger.info("  [推理链]")
                                for _, step in reasoning_steps:
                                    self.logger.info(f"    - {step}")
                            # 输出证据
                            evidence = [(k, v) for k, v in sorted(state.known_facts.items()) if k.startswith("evidence")]
                            if evidence:
                                self.logger.info("  [证据]")
                                for _, ev in evidence:
                                    self.logger.info(f"    - {ev[:100]}..." if len(ev) > 100 else f"    - {ev}")
                        break
                
                # 2.5 评估
                eval_result = self.evaluator.evaluate(
                    question=question,
                    collected_memories=self.context_manager.get_collected_info(),
                    iteration=iteration,
                    reasoning_state=reasoning_state
                )
                
                # 2.5.1 检查问题类型调整建议
                if eval_result.question_type_adjustment and eval_result.question_type_adjustment.should_adjust:
                    new_type_str = eval_result.question_type_adjustment.new_type
                    if new_type_str and reasoning_state:
                        from agent.context.reasoning_cache import QuestionType
                        try:
                            new_type = QuestionType(new_type_str)
                            old_type = reasoning_state.question_type
                            if new_type != old_type:
                                self.logger.info(f"🔄 问题类型调整: {old_type.value} → {new_type.value}")
                                self.logger.info(f"   原因: {eval_result.question_type_adjustment.reason}")
                                # 更新推理状态的问题类型（保留已有的事实和子目标）
                                reasoning_state.question_type = new_type
                                # 根据新类型添加缺失的子目标（如果需要）
                                if new_type == QuestionType.TEMPORAL_REASONING:
                                    # 时序推理需要时间排序相关的子目标
                                    if not any("时间" in g.description or "顺序" in g.description 
                                             for g in reasoning_state.sub_goals):
                                        self.reasoner.cache.add_sub_goal(
                                            description="按时间排序相关事件",
                                            depends_on=[]
                                        )
                        except ValueError:
                            self.logger.warning(f"无效的问题类型: {new_type_str}")
                
                self.context_manager.add_reasoning_step(
                    "evaluate",
                    eval_result.reasoning,
                    {"is_sufficient": eval_result.is_sufficient}
                )
                
                # 2.6 判断是否继续
                if eval_result.is_sufficient:
                    self.logger.info("评估器判断信息已充足")
                    break
                
                # 根据评估结果调整上下文
                if eval_result.memories_to_keep:
                    self.context_manager.prune_memories(eval_result.memories_to_keep)
            
            # 3. 生成总结（如果启用推理）
            summary_result = None
            if enable_reasoning and reasoning_state:
                self.logger.info("生成推理总结...")
                summary_result = self.summarizer.summarize(reasoning_state)
            
            # 4. 构建结果
            execution_time = time.time() - start_time
            
            result = QueryResult(
                retrieved_memories=self.context_manager.build_retrieved_memories(),
                relevant_entities=self.context_manager.get_relevant_entities(),
                relevant_relations=self.context_manager.get_relevant_relations(),
                reasoning_trace=self.context_manager.get_reasoning_trace(),
                total_iterations=iteration,
                total_tool_calls=len(context.tool_results),
                execution_time=execution_time
            )
            
            # 添加推理总结到结果
            if summary_result:
                result.reasoning_trace.append({
                    "type": "summary",
                    "answer": summary_result.answer,
                    "confidence": summary_result.confidence,
                    "reasoning_chain": summary_result.reasoning_chain,
                    "context_text": summary_result.context_text
                })
            
            self.logger.complete(iteration, len(context.tool_results), execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询失败: {str(e)}", e)
            import traceback
            traceback.print_exc()
            return QueryResult(
                reasoning_trace=[{"error": str(e)}],
                execution_time=time.time() - start_time
            )
    
    async def aquery(
        self,
        messages: Union[List[Dict[str, str]], str],
        enable_reasoning: bool = True,
        **kwargs
    ) -> QueryResult:
        """
        异步查询接口
        
        Args:
            messages: OpenAI 格式的消息列表，或直接传入问题字符串
            enable_reasoning: 是否启用推理功能
            **kwargs: 其他参数
            
        Returns:
            QueryResult 对象
        """
        # 标准化输入
        if isinstance(messages, str):
            question = messages
            conversation_history = []
        else:
            question = ""
            conversation_history = []
            for msg in messages:
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                conversation_history.append(msg)
        
        if not question:
            return QueryResult(
                reasoning_trace=[{"error": "未提供问题"}]
            )
        
        start_time = time.time()
        self.logger.start_query(question)
        
        context = self.context_manager.start_query(question)
        
        try:
            # 1. 分析问题
            if enable_reasoning:
                reasoning_state = await self.reasoner.aanalyze_question(question)
            else:
                reasoning_state = None
            
            # 2. ReAct 循环
            iteration = 0
            while iteration < self.config.max_iterations:
                iteration += 1
                self.context_manager.increment_iteration()
                self.logger.iteration(iteration, self.config.max_iterations)
                
                reasoning_state_str = None
                if reasoning_state:
                    reasoning_state_str = self.reasoning_cache.get_state_summary()
                
                # 2.1 异步规划
                plan = await self.planner.aplan(
                    question=question,
                    collected_info=self.context_manager.get_collected_info(),
                    reasoning_state=reasoning_state_str
                )
                
                self.context_manager.add_reasoning_step(
                    "plan",
                    plan.get("analysis", ""),
                    {"tool_calls": [tc.tool_name for tc in plan.get("tool_calls", [])]}
                )
                
                tool_calls = plan.get("tool_calls", [])
                self.logger.plan(plan.get("analysis", "规划中..."), tool_calls)
                
                if plan.get("is_complete", False) or not tool_calls:
                    self.logger.info("规划器判断信息已充足")
                    break
                
                # 2.2 异步执行
                all_results = []
                for executor in self.executors:
                    results = await executor.aexecute(tool_calls)
                    all_results.extend(results)
                    
                    for result in results:
                        self.context_manager.add_tool_result(result.tool_name, result)
                        
                        if reasoning_state and result.data:
                            self.reasoning_cache.record_query(
                                tool_name=result.tool_name,
                                parameters={},
                                iteration=iteration,
                                success=result.is_success,
                                result_summary=result.data.get("message", "") if isinstance(result.data, dict) else ""
                            )
                
                # 2.3 整合事实
                if reasoning_state:
                    self.reasoner.integrate_facts(self.context_manager.get_collected_info())
                
                # 2.4 尝试得出结论
                if reasoning_state:
                    can_conclude, conclusion, confidence = await self.reasoner.atry_conclude()
                    if can_conclude:
                        self.logger.info(f"推理器得出结论（置信度: {confidence:.0%}）")
                        self.logger.info(f"  [结论] {conclusion[:200]}..." if len(conclusion) > 200 else f"  [结论] {conclusion}")
                        # 输出推理依据
                        state = self.reasoner.cache.state
                        if state and state.known_facts:
                            # 输出推理链
                            reasoning_steps = [(k, v) for k, v in sorted(state.known_facts.items()) if k.startswith("reasoning_step")]
                            if reasoning_steps:
                                self.logger.info("  [推理链]")
                                for _, step in reasoning_steps:
                                    self.logger.info(f"    - {step}")
                            # 输出证据
                            evidence = [(k, v) for k, v in sorted(state.known_facts.items()) if k.startswith("evidence")]
                            if evidence:
                                self.logger.info("  [证据]")
                                for _, ev in evidence:
                                    self.logger.info(f"    - {ev[:100]}..." if len(ev) > 100 else f"    - {ev}")
                        break
                
                # 2.5 异步评估
                eval_result = await self.evaluator.aevaluate(
                    question=question,
                    collected_memories=self.context_manager.get_collected_info(),
                    iteration=iteration,
                    reasoning_state=reasoning_state
                )
                
                # 2.5.1 检查问题类型调整建议
                if eval_result.question_type_adjustment and eval_result.question_type_adjustment.should_adjust:
                    new_type_str = eval_result.question_type_adjustment.new_type
                    if new_type_str and reasoning_state:
                        from agent.context.reasoning_cache import QuestionType
                        try:
                            new_type = QuestionType(new_type_str)
                            old_type = reasoning_state.question_type
                            if new_type != old_type:
                                self.logger.info(f"🔄 问题类型调整: {old_type.value} → {new_type.value}")
                                self.logger.info(f"   原因: {eval_result.question_type_adjustment.reason}")
                                # 更新推理状态的问题类型（保留已有的事实和子目标）
                                reasoning_state.question_type = new_type
                                # 根据新类型添加缺失的子目标（如果需要）
                                if new_type == QuestionType.TEMPORAL_REASONING:
                                    # 时序推理需要时间排序相关的子目标
                                    if not any("时间" in g.description or "顺序" in g.description 
                                             for g in reasoning_state.sub_goals):
                                        self.reasoner.cache.add_sub_goal(
                                            description="按时间排序相关事件",
                                            depends_on=[]
                                        )
                        except ValueError:
                            self.logger.warning(f"无效的问题类型: {new_type_str}")
                
                self.context_manager.add_reasoning_step(
                    "evaluate",
                    eval_result.reasoning,
                    {"is_sufficient": eval_result.is_sufficient}
                )
                
                if eval_result.is_sufficient:
                    self.logger.info("评估器判断信息已充足")
                    break
                
                if eval_result.memories_to_keep:
                    self.context_manager.prune_memories(eval_result.memories_to_keep)
            
            # 3. 生成总结
            summary_result = None
            if enable_reasoning and reasoning_state:
                summary_result = await self.summarizer.asummarize(reasoning_state)
            
            # 4. 构建结果
            execution_time = time.time() - start_time
            result = QueryResult(
                retrieved_memories=self.context_manager.build_retrieved_memories(),
                relevant_entities=self.context_manager.get_relevant_entities(),
                relevant_relations=self.context_manager.get_relevant_relations(),
                reasoning_trace=self.context_manager.get_reasoning_trace(),
                total_iterations=iteration,
                total_tool_calls=len(context.tool_results),
                execution_time=execution_time
            )
            
            if summary_result:
                result.reasoning_trace.append({
                    "type": "summary",
                    "answer": summary_result.answer,
                    "confidence": summary_result.confidence,
                    "reasoning_chain": summary_result.reasoning_chain,
                    "context_text": summary_result.context_text
                })
            
            self.logger.complete(iteration, len(context.tool_results), execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询失败: {str(e)}", e)
            return QueryResult(
                reasoning_trace=[{"error": str(e)}],
                execution_time=time.time() - start_time
            )
    
    def get_context_text(self, result: QueryResult) -> str:
        """
        获取用于外部 LLM 的上下文文本
        
        Args:
            result: 查询结果
            
        Returns:
            格式化的上下文文本
        """
        # 优先使用推理总结中的上下文
        for trace in result.reasoning_trace:
            if isinstance(trace, dict) and trace.get("type") == "summary":
                context_text = trace.get("context_text", "")
                if context_text:
                    return context_text
        
        return result.get_context_text()
    
    def get_answer(self, result: QueryResult) -> Optional[str]:
        """
        获取推理得出的答案
        
        Args:
            result: 查询结果
            
        Returns:
            答案（如果有）
        """
        for trace in result.reasoning_trace:
            if isinstance(trace, dict) and trace.get("type") == "summary":
                return trace.get("answer")
        return None
    
    def get_confidence(self, result: QueryResult) -> float:
        """
        获取答案的置信度
        
        Args:
            result: 查询结果
            
        Returns:
            置信度（0-1）
        """
        for trace in result.reasoning_trace:
            if isinstance(trace, dict) and trace.get("type") == "summary":
                return trace.get("confidence", 0.0)
        return 0.0
    
    def add_storage(self, storage_path: str = None, storage_manager: Any = None):
        """
        添加记忆库
        
        Args:
            storage_path: 记忆库路径
            storage_manager: StorageManager 实例
        """
        if storage_manager:
            self.storage_managers.append(storage_manager)
            registry = create_default_registry(storage_manager)
            self.tool_registries.append(registry)
            self.executors.append(
                Executor(
                    tool_registry=registry,
                    parallel=self.config.parallel_tools,
                    timeout=self.config.tool_timeout,
                    logger=self.logger
                )
            )
        elif storage_path:
            self._init_storage_managers([storage_path])
            if self.storage_managers:
                sm = self.storage_managers[-1]
                registry = create_default_registry(sm)
                self.tool_registries.append(registry)
                self.executors.append(
                    Executor(
                        tool_registry=registry,
                        parallel=self.config.parallel_tools,
                        timeout=self.config.tool_timeout,
                        logger=self.logger
                    )
                )
