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
    """Describe how the MCP client should start the Amap MCP subprocess.

    The returned dictionary is not a tool call. It is only a startup recipe that says:
    which Python executable to use, which module to run, which transport to use, and
    which environment variables should be forwarded into that child process.
    """

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
    """Ask the MCP server for its tool list and convert it into LangChain tools.

    This is the first important distinction for beginners:
    this function does not execute weather/geocode queries.
    It only performs tool discovery, so the graph knows which tools exist and how
    each tool should be called later.
    """

    client = MultiServerMCPClient(
        {
            "amap": build_amap_server_config(),
        }
    )
    return await client.get_tools()


async def load_amap_tools():
    """Public async wrapper that returns the discovered MCP tools as a tuple."""

    return tuple(await _load_amap_tools_async())


def _load_amap_tools_in_thread():
    """Load MCP tools in a helper thread when the caller already owns an event loop.

    LangGraph Studio and some web runtimes may already be inside a running asyncio
    event loop. In that case we cannot call ``asyncio.run`` directly in the same
    thread, so this helper moves the blocking bootstrap work to a separate thread.
    """

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
    """Return a cached tuple of MCP-backed tools for the runtime graph.

    The cache matters because MCP tool discovery is relatively expensive:
    it may start a subprocess, initialize the MCP protocol, and request the tool list.
    We do that once per process and then reuse the result.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return tuple(asyncio.run(load_amap_tools()))
    return tuple(_load_amap_tools_in_thread())
