"""
Embedding客户端：支持自定义embedding模型
"""
from typing import List, Optional, Union
import numpy as np


class EmbeddingClient:
    """Embedding客户端 - 支持多种embedding模型"""
    
    def __init__(self, model_path: Optional[str] = None, model_name: Optional[str] = None,
                 device: str = "cpu", use_local: bool = True):
        """
        初始化Embedding客户端
        
        Args:
            model_path: 本地模型路径（优先使用）
            model_name: 模型名称（如果使用HuggingFace模型）
            device: 计算设备 ("cpu" 或 "cuda")
            use_local: 是否优先使用本地模型
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.use_local = use_local
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化embedding模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.model_path and self.use_local:
                # 使用本地模型路径
                print(f"加载本地embedding模型: {self.model_path}")
                self.model = SentenceTransformer(
                    self.model_path,
                    device=self.device,
                    trust_remote_code=True
                )
            elif self.model_name:
                # 使用HuggingFace模型名称
                print(f"加载HuggingFace embedding模型: {self.model_name}")
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True
                )
            else:
                # 使用默认模型
                print("使用默认embedding模型: all-MiniLM-L6-v2")
                self.model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    device=self.device
                )
        except ImportError:
            self.model = None
            print("警告：未安装sentence-transformers库，将使用文本相似度搜索")
            print("安装命令: pip install sentence-transformers")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
        
        Returns:
            向量数组（numpy array）
        """
        if self.model is None:
            # 如果没有embedding模型，返回None
            return None
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"Embedding编码错误: {e}")
            return None
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度
        
        Args:
            query_embedding: 查询向量
            candidate_embeddings: 候选向量数组
        
        Returns:
            相似度数组
        """
        if query_embedding is None or candidate_embeddings is None:
            return None
        
        # 归一化
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-9
        )
        
        # 计算余弦相似度
        similarities = np.dot(candidate_norms, query_norm)
        return similarities
    
    def is_available(self) -> bool:
        """检查embedding模型是否可用"""
        return self.model is not None
