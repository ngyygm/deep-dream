"""
文档处理模块：滑动窗口切分
"""
from typing import List, Tuple

from core.text_chunking import split_markdown_chunks


class DocumentProcessor:
    """文档处理器 - 支持滑动窗口读取"""
    
    def __init__(self, window_size: int = 1000, overlap: int = 200):
        """
        初始化文档处理器
        
        Args:
            window_size: 窗口大小（字符数）
            overlap: 重叠大小（字符数）
        """
        self.window_size = window_size
        self.overlap = overlap

    def chunk_text(self, content: str) -> List[Tuple[str, int, int, str]]:
        """Split Markdown by headings, then paragraph/sentence boundaries.

        Returns a list of 4-tuples: (content, start_offset, end_offset, heading_path).
        """
        chunks = split_markdown_chunks(
            content or "",
            window_size=self.window_size,
            overlap=self.overlap,
        )
        return [
            (
                str(chunk["content"]),
                int(chunk["start_offset"]),
                int(chunk["end_offset"]),
                str(chunk.get("heading_path") or ""),
            )
            for chunk in chunks
        ]
    
