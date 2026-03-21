from langgraph_study.backend.api import get_frontend_dir


def test_frontend_index_contains_expected_assets() -> None:
    frontend_dir = get_frontend_dir()
    html = (frontend_dir / "index.html").read_text(encoding="utf-8")
    script = (frontend_dir / "app.js").read_text(encoding="utf-8")

    assert "Travel Assistant" in html
    assert "sessionList" in html
    assert "chatHistory" in html
    assert "stateQueryContext" in html
    assert "/static/app.js" in html
    assert "/api/chat/stream" in script
    assert "/api/threads" in script
