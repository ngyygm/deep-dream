"""Verify all V1.5 modules are importable with expected public API."""

import importlib


def test_schema_v15_imports():
    mod = importlib.import_module("core.storage.sqlite.schema_v15")
    assert callable(getattr(mod, "init_schema_v15", None))
    assert callable(getattr(mod, "create_tables", None))
    assert callable(getattr(mod, "create_indexes", None))
    assert callable(getattr(mod, "create_fts", None))
    assert callable(getattr(mod, "create_views", None))


def test_integrity_imports():
    mod = importlib.import_module("core.storage.sqlite.integrity")
    assert callable(getattr(mod, "validate_all", None))
    validators = getattr(mod, "ALL_VALIDATORS", None)
    assert validators is not None
    assert len(validators) == 12


def test_content_fs_imports():
    mod = importlib.import_module("core.storage.sqlite.content_fs")
    assert callable(getattr(mod, "compute_content_hash", None))
    assert callable(getattr(mod, "write_version_snapshot", None))
    assert callable(getattr(mod, "write_current_file", None))
    assert callable(getattr(mod, "rebuild_current_files", None))


def test_repos_documents_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.documents")
    for name in ["insert_document", "get_document", "soft_delete_document",
                 "update_current_version", "insert_document_version",
                 "get_active_version", "list_documents"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_repos_episodes_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.episodes")
    for name in ["insert_episode", "get_episode", "fts_sync_episode",
                 "rebuild_fts_all", "supersede_episodes_by_version"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_repos_entities_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.entities")
    for name in ["upsert_entity_family", "insert_entity_observation",
                 "insert_entity_mention", "list_entity_families"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_repos_relations_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.relations")
    for name in ["upsert_relation_family", "insert_relation_assertion",
                 "validate_same_episode", "list_relation_families"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_repos_embeddings_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.embeddings")
    for name in ["insert_embedding", "get_embedding", "vacuum_orphaned",
                 "vacuum_deleted_documents", "count_embeddings"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_repos_search_imports():
    mod = importlib.import_module("core.storage.sqlite.repositories.search")
    for name in ["search_fts", "get_graph_edges", "get_graph_neighbors",
                 "get_document_graph"]:
        assert callable(getattr(mod, name, None)), f"Missing: {name}"


def test_library_manager_imports():
    mod = importlib.import_module("core.storage.sqlite.library_manager")
    assert callable(getattr(mod, "LibraryManager", None))


def test_no_numpy_dependency_in_v15():
    modules = [
        "core.storage.sqlite.schema_v15",
        "core.storage.sqlite.integrity",
        "core.storage.sqlite.content_fs",
        "core.storage.sqlite.repositories.documents",
        "core.storage.sqlite.repositories.episodes",
        "core.storage.sqlite.repositories.entities",
        "core.storage.sqlite.repositories.relations",
        "core.storage.sqlite.repositories.embeddings",
        "core.storage.sqlite.repositories.search",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        source = open(mod.__file__, "r", encoding="utf-8").read()
        assert "import numpy" not in source, f"{mod_name} depends on numpy"
