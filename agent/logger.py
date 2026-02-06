"""
决策链路日志输出

支持三种详细程度：
- minimal: 只显示关键步骤和结果
- moderate: 显示规划、工具调用、结果摘要
- verbose: 显示完整的 LLM 输入输出和推理过程
"""
import sys
from datetime import datetime
from typing import Any, Optional, List
from enum import Enum


class LogLevel(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    VERBOSE = "verbose"


class Colors:
    """终端颜色"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # 前景色
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class AgentLogger:
    """Agent 日志记录器"""
    
    def __init__(self, level: str = "moderate", enable_colors: bool = True):
        self.level = LogLevel(level) if isinstance(level, str) else level
        self.enable_colors = enable_colors and sys.stdout.isatty()
        self._indent = 0
        self._iteration = 0
    
    def _color(self, text: str, color: str) -> str:
        """添加颜色"""
        if self.enable_colors:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def _prefix(self, tag: str, color: str = Colors.WHITE) -> str:
        """生成前缀"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        indent = "  " * self._indent
        colored_tag = self._color(f"[{tag}]", color)
        return f"{self._color(timestamp, Colors.DIM)} {indent}{colored_tag}"
    
    def _print(self, message: str):
        """打印消息"""
        print(message, flush=True)
    
    def set_iteration(self, iteration: int):
        """设置当前迭代次数"""
        self._iteration = iteration
    
    def start_query(self, question: str):
        """开始查询"""
        self._print("")
        self._print(self._color("=" * 60, Colors.CYAN))
        self._print(f"{self._prefix('START', Colors.CYAN + Colors.BOLD)} 开始记忆检索")
        self._print(f"{self._prefix('QUERY', Colors.CYAN)} {self._color(question, Colors.BOLD)}")
        self._print(self._color("=" * 60, Colors.CYAN))
    
    def plan(self, description: str, tool_calls: Optional[List[Any]] = None):
        """规划步骤"""
        self._print(f"{self._prefix('PLAN', Colors.YELLOW)} {description}")
        
        if tool_calls and self.level != LogLevel.MINIMAL:
            self._indent += 1
            for call in tool_calls:
                tool_name = call.tool_name if hasattr(call, 'tool_name') else call.get('tool_name', '')
                params = call.parameters if hasattr(call, 'parameters') else call.get('parameters', {})
                params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                self._print(f"{self._prefix('TOOL', Colors.MAGENTA)} {tool_name}({params_str})")
            self._indent -= 1
    
    def execute(self, tool_name: str, parameters: dict):
        """执行工具"""
        if self.level == LogLevel.MINIMAL:
            return
        
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in parameters.items())
        self._print(f"{self._prefix('EXEC', Colors.BLUE)} {tool_name}({params_str})")
    
    def result(self, tool_name: str, result: Any, success: bool = True):
        """工具结果"""
        if success:
            status = self._color("✓", Colors.GREEN)
        else:
            status = self._color("✗", Colors.RED)
        
        # 结果摘要
        if isinstance(result, list):
            summary = f"返回 {len(result)} 条结果"
        elif isinstance(result, dict):
            if 'error' in result:
                summary = f"错误: {result['error']}"
            else:
                summary = f"返回数据: {list(result.keys())}"
        elif result is None:
            summary = "无结果"
        else:
            summary = str(result)[:100]
        
        self._print(f"{self._prefix('RESULT', Colors.GREEN if success else Colors.RED)} {status} {tool_name}: {summary}")
        
        # verbose 模式显示完整结果
        if self.level == LogLevel.VERBOSE and result:
            self._indent += 1
            if isinstance(result, list):
                for i, item in enumerate(result[:5]):  # 最多显示5条
                    self._print(f"{self._prefix('DATA', Colors.DIM)} [{i}] {str(item)[:200]}")
                if len(result) > 5:
                    self._print(f"{self._prefix('DATA', Colors.DIM)} ... 还有 {len(result) - 5} 条")
            elif isinstance(result, dict):
                for k, v in list(result.items())[:10]:
                    self._print(f"{self._prefix('DATA', Colors.DIM)} {k}: {str(v)[:100]}")
            self._indent -= 1
    
    def evaluate(self, is_sufficient: bool, reasoning: str):
        """评估结果"""
        if is_sufficient:
            status = self._color("✓ 信息充足", Colors.GREEN + Colors.BOLD)
        else:
            status = self._color("→ 需要更多信息", Colors.YELLOW)
        
        self._print(f"{self._prefix('EVAL', Colors.CYAN)} {status}")
        
        if self.level != LogLevel.MINIMAL:
            self._indent += 1
            self._print(f"{self._prefix('REASON', Colors.DIM)} {reasoning[:200]}")
            self._indent -= 1
    
    def iteration(self, iteration: int, max_iterations: int):
        """迭代信息"""
        self._iteration = iteration
        self._print("")
        self._print(f"{self._prefix('ITER', Colors.CYAN)} 第 {iteration}/{max_iterations} 轮迭代")
        self._print(self._color("-" * 40, Colors.DIM))
    
    def context_update(self, kept: int, discarded: int):
        """上下文更新"""
        if self.level == LogLevel.MINIMAL:
            return
        self._print(f"{self._prefix('CTX', Colors.MAGENTA)} 保留 {kept} 条记忆，丢弃 {discarded} 条")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """错误信息"""
        self._print(f"{self._prefix('ERROR', Colors.RED + Colors.BOLD)} {message}")
        if exception and self.level == LogLevel.VERBOSE:
            self._indent += 1
            self._print(f"{self._prefix('TRACE', Colors.RED)} {str(exception)}")
            self._indent -= 1
    
    def warning(self, message: str):
        """警告信息"""
        self._print(f"{self._prefix('WARN', Colors.YELLOW)} {message}")
    
    def info(self, message: str):
        """信息"""
        if self.level != LogLevel.MINIMAL:
            self._print(f"{self._prefix('INFO', Colors.WHITE)} {message}")
    
    def debug(self, message: str):
        """调试信息"""
        if self.level == LogLevel.VERBOSE:
            self._print(f"{self._prefix('DEBUG', Colors.DIM)} {message}")
    
    def llm_input(self, prompt: str):
        """LLM 输入"""
        if self.level == LogLevel.VERBOSE:
            self._print(f"{self._prefix('LLM_IN', Colors.MAGENTA)} Prompt:")
            self._indent += 1
            for line in prompt.split('\n')[:20]:  # 最多显示20行
                self._print(f"{self._prefix('', Colors.DIM)} {line}")
            self._indent -= 1
    
    def llm_output(self, response: str):
        """LLM 输出"""
        if self.level == LogLevel.VERBOSE:
            self._print(f"{self._prefix('LLM_OUT', Colors.GREEN)} Response:")
            self._indent += 1
            for line in response.split('\n')[:20]:
                self._print(f"{self._prefix('', Colors.DIM)} {line}")
            self._indent -= 1
    
    def complete(self, total_iterations: int, total_tool_calls: int, execution_time: float):
        """完成查询"""
        self._print("")
        self._print(self._color("=" * 60, Colors.GREEN))
        self._print(f"{self._prefix('DONE', Colors.GREEN + Colors.BOLD)} 记忆检索完成")
        self._print(f"{self._prefix('STAT', Colors.GREEN)} 迭代次数: {total_iterations}, 工具调用: {total_tool_calls}, 耗时: {execution_time:.2f}s")
        self._print(self._color("=" * 60, Colors.GREEN))
        self._print("")


# 全局默认 logger
_default_logger: Optional[AgentLogger] = None


def get_logger() -> AgentLogger:
    """获取全局 logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = AgentLogger()
    return _default_logger


def set_logger(logger: AgentLogger):
    """设置全局 logger"""
    global _default_logger
    _default_logger = logger
