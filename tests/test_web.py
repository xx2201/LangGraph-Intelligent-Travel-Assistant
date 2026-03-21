from langgraph_study.app.web import render_homepage


def test_render_homepage_contains_expected_title() -> None:
    html = render_homepage()

    assert "LangGraph 旅行助手" in html
    assert "/api/chat" in html
