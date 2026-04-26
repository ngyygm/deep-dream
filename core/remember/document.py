"""
文档处理模块：多文档选择、滑动窗口读取
"""
from typing import List, Iterator, Tuple, Optional
from pathlib import Path

from core.utils import wprint_info


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
    
    def process_documents(self, document_paths: List[str],
                         resume_document_path: Optional[str] = None,
                         resume_text: Optional[str] = None) -> Iterator[Tuple[str, str, bool, int, int, int, str]]:
        """
        处理多个文档，返回滑动窗口迭代器

        Args:
            document_paths: 文档路径列表
            resume_document_path: 断点续传时的起始文档路径（可选）
            resume_text: 断点续传时的起始文本内容（可选，用于定位位置）

        Yields:
            (text_chunk, document_name, is_new_document, start_pos, end_pos, total_length, document_path)
            - text_chunk: 当前窗口的文本内容
            - document_name: 文档名称
            - is_new_document: 是否是新的文档（用于添加提示）
            - start_pos: 当前窗口在文档中的起始位置（字符位置）
            - end_pos: 当前窗口在文档中的结束位置（字符位置）
            - total_length: 文档总长度（字符数）
            - document_path: 文档完整路径
        """
        # 重新排序文档列表，支持断点续传（同时缓存已读文件内容）
        ordered_paths, resume_start_pos, content_cache = self._reorder_documents_for_resume(
            document_paths, resume_document_path, resume_text
        )

        is_first_doc = True
        for doc_path in ordered_paths:
            doc_path_obj = Path(doc_path)
            if not doc_path_obj.exists():
                wprint_info(f"警告：文档不存在: {doc_path}")
                continue

            document_name = doc_path_obj.name

            # 读取文档内容（优先使用缓存，避免断点续传场景的重复读取）
            try:
                content = content_cache.get(doc_path)
                if content is None:
                    with open(doc_path_obj, 'r', encoding='utf-8') as f:
                        content = f.read()
            except Exception as e:
                wprint_info(f"错误：无法读取文档 {doc_path}: {e}")
                continue
            
            total_length = len(content)
            
            # 确定起始位置（仅对第一个文档应用断点续传的起始位置）
            if is_first_doc and resume_start_pos is not None and resume_start_pos > 0:
                start = resume_start_pos
                is_first_doc = False
                wprint_info(f"[断点续传] 从文档 {document_name} 的位置 {start} 继续处理")
            else:
                start = 0
                is_first_doc = False
            
            is_first_chunk = (start == 0)
            # Pre-compute doc prefix outside the loop (invariant per document)
            _doc_prefix = f"[文档元数据] 文档名：{document_name} [/文档元数据]\n\n"
            
            while start < total_length:
                end = min(start + self.window_size, total_length)
                chunk = content[start:end]

                # 如果是新文档的第一块，添加提示
                is_new_doc = is_first_chunk and start == 0
                if is_first_chunk:
                    chunk = _doc_prefix + chunk
                    is_first_chunk = False

                yield (chunk, document_name, is_new_doc, start, end, total_length, doc_path)

                # 移动到下一个窗口（考虑重叠）
                if end >= total_length:
                    break
                start = end - self.overlap
    
    def _reorder_documents_for_resume(self, document_paths: List[str],
                                      resume_document_path: Optional[str],
                                      resume_text: Optional[str]) -> Tuple[List[str], Optional[int], dict]:
        """
        重新排序文档列表以支持断点续传

        优先级：
        1. 如果提供了 resume_document_path，先检查该文档是否在列表中
        2. 如果提供了 resume_text，在所有文档中搜索该文本片段

        Args:
            document_paths: 原始文档路径列表
            resume_document_path: 断点续传时的起始文档路径
            resume_text: 断点续传时的起始文本内容

        Returns:
            (重新排序后的文档路径列表, 在第一个文档中的起始位置, 已缓存的内容字典)
        """
        content_cache: dict = {}

        if not resume_document_path and not resume_text:
            return document_paths, None, content_cache
        
        resume_start_pos = None
        matched_doc_path = None
        
        # 方法1：根据文档路径直接匹配
        if resume_document_path:
            # 尝试精确匹配
            if resume_document_path in document_paths:
                matched_doc_path = resume_document_path
                wprint_info(f"[断点续传] 根据文档路径找到匹配: {resume_document_path}")
            else:
                # 尝试按文件名匹配
                resume_doc_name = Path(resume_document_path).name
                for doc_path in document_paths:
                    if Path(doc_path).name == resume_doc_name:
                        matched_doc_path = doc_path
                        wprint_info(f"[断点续传] 根据文件名找到匹配: {resume_doc_name} -> {doc_path}")
                        break
        
        # 方法2：如果根据文档路径没找到，或者需要定位具体位置，通过文本搜索
        if resume_text:
            # 清理文本（移除窗口添加的前缀）
            search_text = resume_text
            if "开始阅读新的文档，文件名是：" in search_text:
                # 移除旧版前缀，只保留原始文本内容
                parts = search_text.split("\n\n", 1)
                if len(parts) > 1:
                    search_text = parts[1]
            if "[文档元数据]" in search_text:
                # 移除新版元数据标签
                parts = search_text.split("\n\n", 1)
                if len(parts) > 1:
                    search_text = parts[1]
            
            # 在所有文档中搜索该文本片段
            # Optimization: check matched_doc_path first (from path matching), then others
            search_order = []
            if matched_doc_path:
                search_order.append(matched_doc_path)
                search_order.extend(p for p in document_paths if p != matched_doc_path)
            else:
                search_order = list(document_paths)

            # Use a short prefix of search_text for fast pre-filtering
            search_prefix = search_text[:200] if len(search_text) >= 200 else search_text

            for doc_path in search_order:
                doc_path_obj = Path(doc_path)
                if not doc_path_obj.exists():
                    continue

                try:
                    # Read full content once — use first 4KB for quick prefix pre-filter
                    with open(doc_path_obj, 'r', encoding='utf-8') as f:
                        content = f.read()
                    head = content[:4096]

                    # If prefix not in first 4KB, skip full scan for non-matched docs
                    if doc_path != matched_doc_path and head.find(search_prefix) == -1:
                        continue

                    content_cache[doc_path] = content  # cache for reuse

                    # 搜索文本片段的位置
                    text_pos = content.find(search_text)
                    if text_pos != -1:
                        matched_doc_path = doc_path
                        resume_start_pos = text_pos
                        wprint_info(f"[断点续传] 在文档 {doc_path} 中找到匹配文本，位置: {text_pos}")
                        wprint_info(f"[断点续传] 将从位置 {resume_start_pos} 继续处理")
                        break
                except Exception as e:
                    wprint_info(f"警告：无法读取文档 {doc_path}: {e}")
                    continue
        
        if matched_doc_path:
            # 重新排序：将匹配的文档放在第一位
            reordered_paths = [matched_doc_path]
            for doc_path in document_paths:
                if doc_path != matched_doc_path:
                    reordered_paths.append(doc_path)
            return reordered_paths, resume_start_pos, content_cache
        else:
            if resume_document_path or resume_text:
                wprint_info("[断点续传] 警告：未找到匹配的断点位置，将从头开始处理")
            return document_paths, None, content_cache
    
