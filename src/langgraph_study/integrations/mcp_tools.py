from __future__ import annotations

import asyncio
import os
import sys
from queue import Queue
from functools import lru_cache
from threading import Thread

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..core.config import AMAP_MCP_MODULE


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


async def load_amap_tools():
    return tuple(await _load_amap_tools_async())


def _load_amap_tools_in_thread():
    result_queue: Queue = Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put((True, asyncio.run(load_amap_tools())))
        except Exception as exc:  # pragma: no cover - defensive path
            result_queue.put((False, exc))

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    ok, payload = result_queue.get()
    if ok:
        return payload
    raise payload


@lru_cache(maxsize=1)
def get_amap_tools():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return tuple(asyncio.run(load_amap_tools()))
    return tuple(_load_amap_tools_in_thread())
