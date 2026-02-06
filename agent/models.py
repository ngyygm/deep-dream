"""
Agent 数据模型定义
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ToolStatus(Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AgentConfig:
    """Agent 配置"""
    # LLM 配置
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    
    # 记忆库配置
    storage_paths: List[str] = field(default_factory=list)
    
    # 执行配置
    max_iterations: int = 10  # 安全限制，但主要由 LLM 决定何时停止
    parallel_tools: bool = True  # 是否并行执行工具
    tool_timeout: float = 30.0  # 工具执行超时时间（秒）
    
    # 日志配置
    verbose: bool = True  # 是否打印决策链路
    log_level: str = "moderate"  # minimal, moderate, verbose
    
    # 缓存配置
    enable_cache: bool = True  # 是否启用智能缓存
    
    # Embedding 配置（可选，用于语义搜索）
    embedding_model_path: Optional[str] = None
    embedding_device: str = "cpu"


@dataclass
class ToolCall:
    """工具调用"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = ""  # 用于追踪
    
    def __post_init__(self):
        if not self.call_id:
            import uuid
            self.call_id = str(uuid.uuid4())[:8]


@dataclass
class ToolResult:
    """工具执行结果"""
    call_id: str
    tool_name: str
    status: ToolStatus
    data: Any = None
    error_message: str = ""
    execution_time: float = 0.0  # 执行耗时（秒）
    
    @property
    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS


@dataclass
class RetrievedMemory:
    """检索到的记忆"""
    memory_type: str  # "entity", "relation", "cache"
    content: str
    source_id: str  # entity_id 或 relation_id 或 cache_id
    physical_time: Optional[datetime] = None
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """规划步骤"""
    step_id: int
    description: str
    tool_calls: List[ToolCall]
    reasoning: str = ""


@dataclass
class QuestionTypeAdjustment:
    """问题类型调整建议"""
    should_adjust: bool = False
    new_type: Optional[str] = None  # "direct" / "reasoning" / "temporal_reasoning"
    reason: str = ""


@dataclass
class EvaluationResult:
    """评估结果"""
    is_sufficient: bool  # 当前记忆是否足够回答问题
    reasoning: str  # 评估理由
    memories_to_keep: List[str] = field(default_factory=list)  # 保留的记忆 ID
    next_action: str = ""  # 建议的下一步动作（如果不足够）
    question_type_adjustment: Optional[QuestionTypeAdjustment] = None  # 问题类型调整建议


@dataclass
class QueryResult:
    """查询结果"""
    # 检索到的记忆
    retrieved_memories: List[RetrievedMemory] = field(default_factory=list)
    
    # 相关实体
    relevant_entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # 相关关系
    relevant_relations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 推理过程追踪
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    # 元数据
    total_iterations: int = 0
    total_tool_calls: int = 0
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "retrieved_memories": [
                {
                    "type": m.memory_type,
                    "content": m.content,
                    "source_id": m.source_id,
                    "physical_time": m.physical_time.isoformat() if m.physical_time else None,
                    "relevance_score": m.relevance_score,
                    "metadata": m.metadata
                }
                for m in self.retrieved_memories
            ],
            "relevant_entities": self.relevant_entities,
            "relevant_relations": self.relevant_relations,
            "reasoning_trace": self.reasoning_trace,
            "metadata": {
                "total_iterations": self.total_iterations,
                "total_tool_calls": self.total_tool_calls,
                "execution_time": self.execution_time
            }
        }
    
    def get_context_text(self) -> str:
        """获取用于 LLM 上下文的文本"""
        lines = []
        
        if self.relevant_entities:
            lines.append("## 相关实体")
            for entity in self.relevant_entities:
                lines.append(f"- **{entity.get('name', 'Unknown')}**: {entity.get('content', '')}")
        
        if self.relevant_relations:
            lines.append("\n## 相关关系")
            for relation in self.relevant_relations:
                lines.append(f"- {relation.get('content', '')}")
        
        if self.retrieved_memories:
            lines.append("\n## 检索到的记忆")
            for memory in self.retrieved_memories:
                lines.append(f"- [{memory.memory_type}] {memory.content}")
        
        return "\n".join(lines)


@dataclass
class Message:
    """消息（OpenAI 格式）"""
    role: str  # "system", "user", "assistant"
    content: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        return cls(role=data["role"], content=data["content"])
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
