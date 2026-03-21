import os

from langgraph_study.core.config import AMAP_MCP_MODULE
from langgraph_study.integrations.mcp_tools import build_amap_server_config


def test_build_amap_server_config_includes_parent_env(monkeypatch) -> None:
    monkeypatch.setenv("AMAP_API_KEY", "test-amap-key")
    monkeypatch.setenv("GAODE_API_KEY", "test-gaode-key")

    config = build_amap_server_config()

    assert config["transport"] == "stdio"
    assert config["command"] == os.sys.executable
    assert config["args"] == ["-m", AMAP_MCP_MODULE, "--transport", "stdio"]
    assert config["env"] == {
        "AMAP_API_KEY": "test-amap-key",
        "GAODE_API_KEY": "test-gaode-key",
    }
