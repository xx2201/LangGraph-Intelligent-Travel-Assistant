from __future__ import annotations

DEFAULT_USER_INPUT = "帮我看看北京今天天气怎么样，适合出门吗？"
DEFAULT_QWEN_MODEL = "qwen3-max"
AMAP_MCP_MODULE = "langgraph_study.amap_mcp_server"
CHECKPOINT_DB_PATH = ".langgraph_data/checkpoints.sqlite"
TRAVEL_AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手，负责回答天气、地点、城市出行、景点、地址解析等问题。

行为规则：
1. 只要问题涉及天气、地点、坐标、行政区、POI、出行前准备，就优先调用高德 MCP 工具。
2. 不要假装知道实时天气或地点信息；没有工具结果时要明确说明。
3. 回答使用中文，先给直接结论，再给简洁解释。
4. 如果用户问题与旅行无关，也可以正常回答，但保持简洁。
5. 如果工具返回的数据不足以支撑结论，要说明不确定性。
""".strip()
