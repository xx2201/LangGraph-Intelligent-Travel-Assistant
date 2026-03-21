from langgraph_study.backend.api import chunk_text, extract_stream_text


def test_chunk_text_splits_reply_into_small_pieces() -> None:
    chunks = chunk_text("abcdef", chunk_size=2)

    assert chunks == ["ab", "cd", "ef"]


def test_extract_stream_text_handles_plain_string_content() -> None:
    class FakeChunk:
        content = "hello"

    assert extract_stream_text(FakeChunk()) == "hello"


def test_extract_stream_text_handles_list_content() -> None:
    class FakeChunk:
        content = [{"text": "hel"}, {"text": "lo"}]

    assert extract_stream_text(FakeChunk()) == "hello"
