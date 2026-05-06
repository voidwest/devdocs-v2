from app.ingest import chunk_text_stream


def test_chunk_text_stream_single_chunk():
    text = ["hello world"]
    chunks = list(chunk_text_stream(text, chunk_size=20, overlap=5))
    assert chunks == ["hello world"]


def test_chunk_text_stream_splits_text():
    text = ["abcdefghijklmnopqrstuvwxyz"]
    chunks = list(chunk_text_stream(text, chunk_size=5, overlap=0))
    assert chunks == ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]


def test_chunk_text_stream_with_overlap():
    text = ["0123456789"]
    chunks = list(chunk_text_stream(text, chunk_size=5, overlap=2))
    assert chunks == ["01234", "34567", "6789"]


def test_chunk_text_stream_multiple_lines():
    text = ["line one", "line two", "line three"]
    chunks = list(chunk_text_stream(text, chunk_size=10, overlap=0))
    assert chunks[0].startswith("line one\n")


def test_chunk_text_stream_empty_input():
    chunks = list(chunk_text_stream([""], chunk_size=5, overlap=0))
    assert chunks == []
