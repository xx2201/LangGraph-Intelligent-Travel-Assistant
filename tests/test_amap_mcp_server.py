from typing import Any

from langgraph_study.mcp import amap_server


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_request_amap_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("AMAP_API_KEY", raising=False)
    monkeypatch.delenv("GAODE_API_KEY", raising=False)

    try:
        amap_server._request_amap("weather/weatherInfo", {"city": "北京"})
    except RuntimeError as exc:
        assert "AMAP_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when API key is missing")


def test_geocode_returns_structured_payload(monkeypatch) -> None:
    monkeypatch.setattr(amap_server, "_request_amap", lambda path, params: {
        "status": "1",
        "info": "OK",
        "infocode": "10000",
        "data": {
            "count": "1",
            "geocodes": [{"location": "116.48,39.99"}],
        },
    })

    result = amap_server.geocode("北京市朝阳区")

    assert result["status"] == "1"
    assert result["count"] == "1"
    assert result["geocodes"][0]["location"] == "116.48,39.99"


def test_weather_returns_lives(monkeypatch) -> None:
    monkeypatch.setattr(amap_server, "_request_amap", lambda path, params: {
        "status": "1",
        "info": "OK",
        "infocode": "10000",
        "data": {
            "lives": [{"city": "北京", "weather": "晴"}],
            "forecasts": [],
        },
    })

    result = amap_server.weather("北京")

    assert result["status"] == "1"
    assert result["lives"][0]["city"] == "北京"
