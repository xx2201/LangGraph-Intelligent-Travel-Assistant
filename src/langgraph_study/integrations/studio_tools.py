from __future__ import annotations

from langchain_core.tools import tool


def _studio_only_message(tool_name: str) -> str:
    return (
        f"{tool_name} 是 Studio 调试占位工具，当前只用于展示 LangGraph 的图结构和工具调用路径。"
        "如果你需要真实的高德 MCP 查询，请使用 CLI 运行时图。"
    )


@tool
def geocode(address: str, city: str | None = None) -> str:
    """Convert a human-readable address into coordinates."""
    city_hint = f"，city={city}" if city else ""
    return _studio_only_message(f"geocode(address={address}{city_hint})")


@tool
def reverse_geocode(location: str, radius: int = 1000, extensions: str = "base") -> str:
    """Convert coordinates into a structured address."""
    return _studio_only_message(
        f"reverse_geocode(location={location}, radius={radius}, extensions={extensions})"
    )


@tool
def weather(city: str, extensions: str = "base") -> str:
    """Fetch weather information for a city."""
    return _studio_only_message(f"weather(city={city}, extensions={extensions})")


@tool
def input_tips(keywords: str, city: str | None = None, city_limit: bool = False) -> str:
    """Return location suggestions for partial user input."""
    city_hint = f", city={city}" if city else ""
    return _studio_only_message(
        f"input_tips(keywords={keywords}{city_hint}, city_limit={city_limit})"
    )


def get_studio_tools():
    return (geocode, reverse_geocode, weather, input_tips)
