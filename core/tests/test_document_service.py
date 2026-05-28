from pathlib import Path

from core.documents import DocumentService
from core.storage.sqlite import SQLiteGraphStorageManager


def _store(tmp_path: Path) -> SQLiteGraphStorageManager:
    return SQLiteGraphStorageManager(
        storage_path=str(tmp_path / "graphs" / "g"),
        graph_id="g",
        vector_dim=8,
    )


def test_document_service_maps_searches_reads_and_shapes_tree(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("# Note\n\nAlpha line\n\n## Detail\nBeta concept line\n", encoding="utf-8")

    store = _store(tmp_path)
    store.index_vault(str(vault))
    service = DocumentService(store)

    mapped = service.map_path(str(note))
    assert mapped["total"] == 1
    doc = mapped["documents"][0]
    assert doc["source_mode"] == "external"
    assert doc["resolved_path"] == str(note.resolve())

    searched = service.search_files("Beta")
    assert searched["used"]["raw_files"] is True
    assert searched["total"] == 1
    assert searched["hits"][0]["document"]["line_start"] == 6

    content = service.read_document(doc["document_version_id"])
    assert "Beta concept line" in content["content"]
    assert content["source_mode"] == "external"

    tree = service.vault_tree()
    assert tree["total"] == 1
    assert tree["vaults"][0]["vault_root"] == str(vault)
    assert tree["vaults"][0]["files"][0]["relative_path"] == "note.md"
