from .manager import StorageManager
from .embedding import EmbeddingClient


def create_storage_manager(config: dict, embedding_client=None, storage_path=None, **kwargs):
    """根据配置创建存储管理器（工厂模式）。

    Args:
        config: 服务配置字典
        embedding_client: EmbeddingClient 实例（可选）
        storage_path: 显式指定存储路径（优先于 config["storage_path"]）
        **kwargs: 传递给 StorageManager 的额外参数

    Returns:
        StorageManager 或 Neo4jStorageManager 实例
    """
    storage_config = config.get("storage") or {}
    backend = storage_config.get("backend", "sqlite")
    sp = storage_path or config.get("storage_path", "./graph")

    if backend == "neo4j":
        from .neo4j_store import Neo4jStorageManager
        neo4j_config = storage_config.get("neo4j") or {}
        return Neo4jStorageManager(
            storage_path=sp,
            neo4j_uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            neo4j_auth=(
                neo4j_config.get("user", "neo4j"),
                neo4j_config.get("password", "password"),
            ),
            embedding_client=embedding_client,
            entity_content_snippet_length=kwargs.get("entity_content_snippet_length", 50),
            relation_content_snippet_length=kwargs.get("relation_content_snippet_length", 50),
            vector_dim=storage_config.get("vector_dim", 1024),
        )

    # 默认 SQLite 后端
    return StorageManager(
        storage_path=sp,
        embedding_client=embedding_client,
        entity_content_snippet_length=kwargs.get("entity_content_snippet_length", 50),
        relation_content_snippet_length=kwargs.get("relation_content_snippet_length", 50),
    )
