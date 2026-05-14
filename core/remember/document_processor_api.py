"""
Document-level processing API (``process_documents``).

Extracted from orchestrator.py.  The function receives the processor instance
(``TemporalMemoryGraphProcessor``) as the first argument so it can call mixin
methods and access attributes without circular imports.
"""
from typing import List, Optional

from core.remember.entity import EntityProcessor
from core.utils import wprint_info


def process_documents(
    processor,
    document_paths: List[str],
    verbose: bool = True,
    entity_progress_verbose: Optional[bool] = None,
    similarity_threshold: Optional[float] = None,
    max_similar_entities: Optional[int] = None,
    content_snippet_length: Optional[int] = None,
    relation_content_snippet_length: Optional[int] = None,
    load_cache_memory: Optional[bool] = None,
    jaccard_search_threshold: Optional[float] = None,
    embedding_name_search_threshold: Optional[float] = None,
    embedding_full_search_threshold: Optional[float] = None,
):
    """
    Process multiple documents.

    Args:
        document_paths: List of document file paths.
        verbose: Whether to print detailed information.
        entity_progress_verbose: Whether to print entity-alignment progress (default same as verbose).
        similarity_threshold: Entity search similarity threshold (optional, overrides init).
        max_similar_entities: Max similar entities after embedding pre-filter (optional).
        content_snippet_length: Entity content snippet length (optional).
        relation_content_snippet_length: Relation content snippet length (optional).
        load_cache_memory: Whether to load cached memory (optional).
        jaccard_search_threshold: Jaccard search threshold (optional).
        embedding_name_search_threshold: Embedding name search threshold (optional).
        embedding_full_search_threshold: Embedding full search threshold (optional).
    """
    # Save original values for restoration at the end.
    original_values = {}
    original_components = {}
    _original_sub_attrs = {}

    if similarity_threshold is not None:
        original_values['similarity_threshold'] = processor.similarity_threshold
        processor.similarity_threshold = similarity_threshold

    if jaccard_search_threshold is not None:
        original_values['jaccard_search_threshold'] = processor.jaccard_search_threshold
        processor.jaccard_search_threshold = jaccard_search_threshold
    if embedding_name_search_threshold is not None:
        original_values['embedding_name_search_threshold'] = processor.embedding_name_search_threshold
        processor.embedding_name_search_threshold = embedding_name_search_threshold
    if embedding_full_search_threshold is not None:
        original_values['embedding_full_search_threshold'] = processor.embedding_full_search_threshold
        processor.embedding_full_search_threshold = embedding_full_search_threshold

    need_update_entity_processor = False
    final_max_similar_entities = processor.max_similar_entities
    final_content_snippet_length = processor.content_snippet_length

    if max_similar_entities is not None:
        original_values['max_similar_entities'] = processor.max_similar_entities
        processor.max_similar_entities = max_similar_entities
        final_max_similar_entities = max_similar_entities
        need_update_entity_processor = True

    if content_snippet_length is not None:
        original_values['content_snippet_length'] = processor.content_snippet_length
        processor.content_snippet_length = content_snippet_length
        final_content_snippet_length = content_snippet_length
        if 'storage.entity_content_snippet_length' not in _original_sub_attrs:
            _original_sub_attrs['storage.entity_content_snippet_length'] = processor.storage.entity_content_snippet_length
        if 'llm_client.content_snippet_length' not in _original_sub_attrs:
            _original_sub_attrs['llm_client.content_snippet_length'] = processor.llm_client.content_snippet_length
        processor.storage.entity_content_snippet_length = content_snippet_length
        processor.llm_client.content_snippet_length = content_snippet_length
        need_update_entity_processor = True

    if need_update_entity_processor:
        if 'entity_processor' not in original_components:
            original_components['entity_processor'] = processor.entity_processor
        processor.entity_processor = EntityProcessor(
            processor.storage,
            processor.llm_client,
            max_similar_entities=final_max_similar_entities,
            content_snippet_length=final_content_snippet_length,
        )
    if relation_content_snippet_length is not None:
        original_values['relation_content_snippet_length'] = processor.relation_content_snippet_length
        processor.relation_content_snippet_length = relation_content_snippet_length
        if 'storage.relation_content_snippet_length' not in _original_sub_attrs:
            _original_sub_attrs['storage.relation_content_snippet_length'] = processor.storage.relation_content_snippet_length
        processor.storage.relation_content_snippet_length = relation_content_snippet_length
    if load_cache_memory is not None:
        original_values['load_cache_memory'] = processor.load_cache_memory
        processor.load_cache_memory = load_cache_memory

    _saved_entity_progress_verbose = processor.entity_processor.entity_progress_verbose
    _epv = entity_progress_verbose if entity_progress_verbose is not None else verbose
    try:
        processor.entity_processor.entity_progress_verbose = _epv
        if verbose:
            wprint_info(f"开始处理 {len(document_paths)} 个文档...")

        # Resume-from-breakpoint variables
        resume_document_path = None
        resume_text = None

        if processor.load_cache_memory:
            if verbose:
                wprint_info("正在加载最新的缓存记忆...")

            latest_metadata = processor.storage.get_latest_episode_metadata(activity_type="文档处理")

            if latest_metadata:
                processor.current_episode = processor.storage.load_episode(latest_metadata['absolute_id'])

                if processor.current_episode:
                    if verbose:
                        wprint_info(f"已加载缓存记忆: {processor.current_episode.absolute_id} (时间: {processor.current_episode.event_time})")

                    resume_document_path = latest_metadata.get('document_path', '')
                    resume_text = latest_metadata.get('text', '')

                    if verbose:
                        if resume_document_path:
                            wprint_info(f"[断点续传] 上次处理的文档: {resume_document_path}")
                        if resume_text:
                            text_preview = resume_text[:100].replace('\n', ' ')
                            wprint_info(f"[断点续传] 上次处理的文本片段: {text_preview}...")
                else:
                    if verbose:
                        wprint_info("未找到缓存记忆，将从头开始处理")
                    processor.current_episode = None
            else:
                if verbose:
                    wprint_info("不加载缓存记忆，将从头开始处理")
                processor.current_episode = None
        else:
            if verbose:
                wprint_info("不加载缓存记忆，将从头开始处理")
            processor.current_episode = None

        # Iterate all document windows (supports resume-from-breakpoint)
        for chunk_idx, (input_text, document_name, is_new_document, text_start_pos, text_end_pos, total_text_length, document_path) in enumerate(
            processor.document_processor.process_documents(
                document_paths,
                resume_document_path=resume_document_path,
                resume_text=resume_text,
            )
        ):
            if verbose:
                wprint_info(f"\n处理窗口 {chunk_idx + 1} (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
            elif _epv:
                wprint_info(f"窗口 {chunk_idx + 1} 开始 · {document_name}")

            processor._process_window(input_text, document_name, is_new_document,
                                text_start_pos, text_end_pos, total_text_length, verbose,
                                verbose_steps=_epv, document_path=document_path)
    finally:
        for key, value in original_values.items():
            setattr(processor, key, value)
        for key, value in original_components.items():
            setattr(processor, key, value)
        for attr_path, value in _original_sub_attrs.items():
            obj_name, attr_name = attr_path.split('.', 1)
            setattr(getattr(processor, obj_name), attr_name, value)
        processor.entity_processor.entity_progress_verbose = _saved_entity_progress_verbose
