from app.query import build_prompt, trim_context


def test_trim_context_no_trim_needed():
    docs = ["short doc", "another short doc"]
    result = trim_context(docs, max_chars=1000)
    assert result == docs


def test_trim_context_partial_trim():
    docs = ["a" * 50, "b" * 200, "c" * 200]
    result = trim_context(docs, max_chars=100)
    assert len(result) == 1
    assert result == ["a" * 50]


def test_trim_context_skip_tiny_tail():
    docs = ["a" * 50, "b" * 200]
    result = trim_context(docs, max_chars=51)
    assert len(result) == 1
    assert result == ["a" * 50]


def test_trim_context_exact_fit():
    docs = ["a" * 50, "b" * 50]
    result = trim_context(docs, max_chars=100)
    assert len(result) == 2
    assert result == docs


def test_trim_context_large_remaining_trims():
    docs = ["a" * 50, "b" * 200]
    result = trim_context(docs, max_chars=200)
    assert len(result) == 2
    assert len(result[0]) == 50
    assert len(result[1]) == 150


def test_build_prompt_contains_user_query():
    prompt = build_prompt("what is AI?", "some context")
    assert "what is AI?" in prompt
    assert "some context" in prompt
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt


def test_build_prompt_no_empty_sections():
    prompt = build_prompt("test query", "test context")
    assert prompt.strip() != ""
    assert prompt.startswith("<|system|>")
    assert prompt.endswith("<|assistant|>\n")
