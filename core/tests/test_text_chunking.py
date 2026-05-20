from core.remember.document import DocumentProcessor
from core.server.routes.remember import _uploaded_file_to_markdown
from core.storage.sqlite import SQLiteGraphStorageManager
from core.text_chunking import split_markdown_chunks


def test_markdown_chunking_prefers_sentence_boundaries():
    sentence = "第一段说明Alpha和Beta之间的关系。"
    text = "# Long\n" + (sentence * 12) + "\n\n## Next\n尾段。"

    chunks = split_markdown_chunks(text, window_size=80, overlap=10)

    assert len(chunks) > 2
    for chunk in chunks:
        start = chunk["start_offset"]
        end = chunk["end_offset"]
        assert chunk["content"] == text[start:end]
    for chunk in chunks[:-1]:
        content = chunk["content"].rstrip()
        assert content.endswith(("。", "\n"))


def test_document_processor_and_vault_index_share_chunker():
    text = "# Long\n" + ("Alice knows Bob. " * 30)
    processor = DocumentProcessor(window_size=90, overlap=15)

    processor_chunks = processor.chunk_text(text)
    storage_chunks = SQLiteGraphStorageManager.split_markdown_episodes(text, window_size=90, overlap=15)

    assert [(c, s, e) for c, s, e in processor_chunks] == [
        (chunk["content"], chunk["start_offset"], chunk["end_offset"])
        for chunk in storage_chunks
    ]


def test_txt_upload_accepts_utf16_text():
    data = "爱情心理学\nAlice 和 Bob".encode("utf-16")
    markdown = _uploaded_file_to_markdown(data, "爱情心理学.txt", ".txt")

    assert markdown.startswith("# 爱情心理学.txt")
    assert "爱情心理学" in markdown
    assert "Alice 和 Bob" in markdown


def test_txt_upload_accepts_ansi_gbk_text():
    data = "爱情心理学\n黄维仁博士".encode("gbk")
    markdown = _uploaded_file_to_markdown(data, "ansi.txt", ".txt")

    assert markdown.startswith("# ansi.txt")
    assert "爱情心理学" in markdown
    assert "黄维仁博士" in markdown
