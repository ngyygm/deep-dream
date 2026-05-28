from .embedding import EmbeddingClient


def create_storage_manager(config: dict, embedding_client=None, storage_path=None, **kwargs):
    """创建存储管理器。

    使用 V1.5 LibraryManager（SQLite + FTS5）。

    Args:
        config: 服务配置字典
        embedding_client: EmbeddingClient 实例（可选）
        storage_path: 显式指定存储路径（优先于 config["storage_path"]）
        **kwargs: 传递给存储管理器的额外参数

    Returns:
        LibraryManager 实例
    """
    storage_config = config.get("storage") or {}
    sp = storage_path or config.get("storage_path", "./library")

    from .sqlite.library_manager import LibraryManager

    return LibraryManager(
        library_path=sp,
        embedding_client=embedding_client,
        entity_content_snippet_length=kwargs.get("entity_content_snippet_length", 50),
        relation_content_snippet_length=kwargs.get("relation_content_snippet_length", 50),
    )
