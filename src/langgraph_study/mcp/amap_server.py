from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

AMAP_BASE_URL = "https://restapi.amap.com/v3"
mcp = FastMCP("Amap MCP Server", json_response=True)


def _get_amap_api_key() -> str:
    """Read the Amap API key from environment variables."""

    api_key = os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到高德 API Key。请设置 AMAP_API_KEY 或 GAODE_API_KEY 环境变量。"
        )
    return api_key


def _request_amap(
    path: str,
    params: dict[str, Any],
    http_get=requests.get,
) -> dict[str, Any]:
    """Send one HTTP request to the Amap REST API and normalize the response shape."""

    request_params = {
        "key": _get_amap_api_key(),
        **params,
    }
    response = http_get(
        f"{AMAP_BASE_URL}/{path}",
        params=request_params,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "status": data.get("status"),
        "info": data.get("info"),
        "infocode": data.get("infocode"),
        "data": data,
    }


@mcp.tool()
def geocode(address: str, city: str | None = None) -> dict[str, Any]:
    """将结构化地址转换为经纬度坐标。"""
    params: dict[str, Any] = {"address": address}
    if city:
        params["city"] = city
    result = _request_amap("geocode/geo", params)
    geocodes = result["data"].get("geocodes", [])
    return {
        "query": {"address": address, "city": city},
        "count": result["data"].get("count"),
        "geocodes": geocodes,
        "status": result["status"],
        "info": result["info"],
    }


@mcp.tool()
def reverse_geocode(
    location: str,
    radius: int = 1000,
    extensions: str = "base",
) -> dict[str, Any]:
    """将经纬度坐标转换为结构化地址。location 格式应为 'lng,lat'。"""
    result = _request_amap(
        "geocode/regeo",
        {
            "location": location,
            "radius": radius,
            "extensions": extensions,
        },
    )
    return {
        "query": {
            "location": location,
            "radius": radius,
            "extensions": extensions,
        },
        "regeocode": result["data"].get("regeocode", {}),
        "status": result["status"],
        "info": result["info"],
    }


@mcp.tool()
def weather(city: str, extensions: str = "base") -> dict[str, Any]:
    """查询城市天气。city 可传城市名、adcode 或 citycode。"""
    result = _request_amap(
        "weather/weatherInfo",
        {
            "city": city,
            "extensions": extensions,
        },
    )
    return {
        "query": {"city": city, "extensions": extensions},
        "lives": result["data"].get("lives", []),
        "forecasts": result["data"].get("forecasts", []),
        "status": result["status"],
        "info": result["info"],
    }


@mcp.tool()
def input_tips(
    keywords: str,
    city: str | None = None,
    city_limit: bool = False,
) -> dict[str, Any]:
    """根据关键字获取输入提示，适合地点补全。"""
    params: dict[str, Any] = {
        "keywords": keywords,
        "citylimit": "true" if city_limit else "false",
    }
    if city:
        params["city"] = city
    result = _request_amap("assistant/inputtips", params)
    return {
        "query": {
            "keywords": keywords,
            "city": city,
            "city_limit": city_limit,
        },
        "tips": result["data"].get("tips", []),
        "status": result["status"],
        "info": result["info"],
    }


def parse_args() -> argparse.Namespace:
    """Parse the transport mode used when the MCP server process starts."""

    parser = argparse.ArgumentParser(description="Run the Amap MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport type.",
    )
    return parser.parse_args()


def main() -> None:
    """Start the MCP server with the chosen transport."""

    args = parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
