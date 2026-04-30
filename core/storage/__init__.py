from .embedding import EmbeddingClient


def create_storage_manager(config: dict, embedding_client=None, storage_path=None, **kwargs):
    """创建 Neo4j 存储管理器。

    Args:
        config: 服务配置字典
        embedding_client: EmbeddingClient 实例（可选）
        storage_path: 显式指定存储路径（优先于 config["storage_path"]）
        **kwargs: 传递给 Neo4jStorageManager 的额外参数

    Returns:
        Neo4jStorageManager 实例
    """
    from .neo4j_store import Neo4jStorageManager

    storage_config = config.get("storage") or {}
    neo4j_config = storage_config.get("neo4j") or {}
    sp = storage_path or config.get("storage_path", "./graph")

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
        graph_id=kwargs.get("graph_id", "default"),
    )
