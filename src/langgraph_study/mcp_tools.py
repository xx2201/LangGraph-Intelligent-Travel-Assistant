from __future__ import annotations

import asyncio
import os
import sys
from functools import lru_cache

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import AMAP_MCP_MODULE


def build_amap_server_config() -> dict:
    env = {
        key: value
        for key in ("AMAP_API_KEY", "GAODE_API_KEY")
        if (value := os.getenv(key))
    }

    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", AMAP_MCP_MODULE, "--transport", "stdio"],
        "env": env,
    }


async def _load_amap_tools_async():
    client = MultiServerMCPClient(
        {
            "amap": build_amap_server_config(),
        }
    )
    return await client.get_tools()


@lru_cache(maxsize=1)
def get_amap_tools():
    return tuple(asyncio.run(_load_amap_tools_async()))
