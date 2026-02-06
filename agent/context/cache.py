"""
智能缓存

用于缓存查询结果，加速相似查询
"""
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class SmartCache:
    """
    智能缓存
    
    特点：
    - 基于查询参数的哈希键
    - 支持 TTL 过期
    - LLM 可判断是否复用缓存
    """
    
    def __init__(
        self,
        max_size: int = 100,
        default_ttl: int = 3600  # 默认 1 小时
    ):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
    
    def _generate_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 规范化参数
        sorted_params = json.dumps(parameters, sort_keys=True)
        key_str = f"{tool_name}:{sorted_params}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            缓存的结果，如果不存在或已过期则返回 None
        """
        key = self._generate_key(tool_name, parameters)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired:
            del self._cache[key]
            return None
        
        entry.hit_count += 1
        return entry.value
    
    def set(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        设置缓存
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            value: 结果值
            ttl: 过期时间（秒），None 使用默认值
        """
        # 清理过期和超量条目
        self._cleanup()
        
        key = self._generate_key(tool_name, parameters)
        ttl = ttl or self.default_ttl
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl)
        )
    
    def _cleanup(self):
        """清理过期和超量条目"""
        # 删除过期条目
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]
        
        # 如果仍然超量，删除最少使用的条目
        if len(self._cache) >= self.max_size:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at)
            )
            # 删除 20% 的条目
            to_remove = max(1, len(sorted_entries) // 5)
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(entry.hit_count for entry in self._cache.values())
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "entries": [
                {
                    "key": entry.key[:16],
                    "hits": entry.hit_count,
                    "age_seconds": (datetime.now() - entry.created_at).total_seconds()
                }
                for entry in self._cache.values()
            ]
        }
    
    def should_use_cache(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: str = ""
    ) -> bool:
        """
        判断是否应该使用缓存（简单规则版本）
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            context: 上下文（用于更复杂的判断）
            
        Returns:
            是否应该使用缓存
        """
        # 简单规则：如果缓存存在且未过期，就使用
        cached = self.get(tool_name, parameters)
        return cached is not None
