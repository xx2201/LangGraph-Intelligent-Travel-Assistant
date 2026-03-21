from __future__ import annotations

from .mcp.amap_server import (
    geocode,
    input_tips,
    main,
    mcp,
    parse_args,
    reverse_geocode,
    weather,
    _get_amap_api_key,
    _request_amap,
)

__all__ = [
    "_get_amap_api_key",
    "_request_amap",
    "geocode",
    "input_tips",
    "main",
    "mcp",
    "parse_args",
    "reverse_geocode",
    "weather",
]


if __name__ == "__main__":
    main()
