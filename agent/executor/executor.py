"""
Executor 执行器

负责执行工具调用，支持并行执行
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import ToolCall, ToolResult, ToolStatus
from ..logger import AgentLogger, get_logger
from .tool_registry import ToolRegistry


class Executor:
    """执行器 - 执行工具调用"""
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        parallel: bool = True,
        max_workers: int = 5,
        timeout: float = 30.0,
        logger: Optional[AgentLogger] = None
    ):
        """
        初始化执行器
        
        Args:
            tool_registry: 工具注册表
            parallel: 是否并行执行
            max_workers: 最大并行工作数
            timeout: 工具执行超时时间（秒）
            logger: 日志记录器
        """
        self.tool_registry = tool_registry
        self.parallel = parallel
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = logger or get_logger()
    
    def execute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            执行结果列表
        """
        if not tool_calls:
            return []
        
        if self.parallel and len(tool_calls) > 1:
            return self._execute_parallel(tool_calls)
        else:
            return self._execute_sequential(tool_calls)
    
    async def aexecute(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        异步执行工具调用
        
        Args:
            tool_calls: 工具调用列表
            
        Returns:
            执行结果列表
        """
        if not tool_calls:
            return []
        
        if self.parallel and len(tool_calls) > 1:
            return await self._aexecute_parallel(tool_calls)
        else:
            return await self._aexecute_sequential(tool_calls)
    
    def _execute_single(self, tool_call: ToolCall) -> ToolResult:
        """执行单个工具调用"""
        start_time = time.time()
        
        # 获取工具实例
        tool = self.tool_registry.get_tool_instance(tool_call.tool_name)
        
        if not tool:
            self.logger.error(f"Tool not found: {tool_call.tool_name}")
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error_message=f"工具不存在: {tool_call.tool_name}",
                execution_time=time.time() - start_time
            )
        
        try:
            # 记录执行
            self.logger.execute(tool_call.tool_name, tool_call.parameters)
            
            # 执行工具
            result_data = tool.execute(**tool_call.parameters)
            execution_time = time.time() - start_time
            
            # 判断成功与否
            success = result_data.get("success", True) if isinstance(result_data, dict) else True
            
            # 记录结果
            self.logger.result(tool_call.tool_name, result_data, success=success)
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                data=result_data,
                error_message=result_data.get("message", "") if not success else "",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool execution failed: {tool_call.tool_name}", e)
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _aexecute_single(self, tool_call: ToolCall) -> ToolResult:
        """异步执行单个工具调用"""
        start_time = time.time()
        
        tool = self.tool_registry.get_tool_instance(tool_call.tool_name)
        
        if not tool:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error_message=f"工具不存在: {tool_call.tool_name}",
                execution_time=time.time() - start_time
            )
        
        try:
            self.logger.execute(tool_call.tool_name, tool_call.parameters)
            
            # 异步执行
            result_data = await tool.aexecute(**tool_call.parameters)
            execution_time = time.time() - start_time
            
            success = result_data.get("success", True) if isinstance(result_data, dict) else True
            self.logger.result(tool_call.tool_name, result_data, success=success)
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.SUCCESS if success else ToolStatus.ERROR,
                data=result_data,
                error_message=result_data.get("message", "") if not success else "",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool execution failed: {tool_call.tool_name}", e)
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                status=ToolStatus.ERROR,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _execute_sequential(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """顺序执行工具调用"""
        results = []
        for tool_call in tool_calls:
            result = self._execute_single(tool_call)
            results.append(result)
        return results
    
    def _execute_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """并行执行工具调用"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_call = {
                executor.submit(self._execute_single, call): call
                for call in tool_calls
            }
            
            # 收集结果
            for future in as_completed(future_to_call, timeout=self.timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    call = future_to_call[future]
                    results.append(ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status=ToolStatus.TIMEOUT,
                        error_message=f"执行超时或异常: {str(e)}",
                        execution_time=self.timeout
                    ))
        
        # 按原始顺序排序
        call_id_order = {call.call_id: i for i, call in enumerate(tool_calls)}
        results.sort(key=lambda r: call_id_order.get(r.call_id, 999))
        
        return results
    
    async def _aexecute_sequential(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """异步顺序执行"""
        results = []
        for tool_call in tool_calls:
            result = await self._aexecute_single(tool_call)
            results.append(result)
        return results
    
    async def _aexecute_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """异步并行执行"""
        tasks = [
            asyncio.wait_for(
                self._aexecute_single(call),
                timeout=self.timeout
            )
            for call in tool_calls
        ]
        
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
            except asyncio.TimeoutError:
                call = tool_calls[i]
                results.append(ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status=ToolStatus.TIMEOUT,
                    error_message="执行超时",
                    execution_time=self.timeout
                ))
            except Exception as e:
                call = tool_calls[i]
                results.append(ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status=ToolStatus.ERROR,
                    error_message=str(e),
                    execution_time=0
                ))
        
        return results
