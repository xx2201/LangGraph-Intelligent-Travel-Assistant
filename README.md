# LangGraph 智能旅行助手

这是一个基于 LangGraph、`qwen3-max` 和高德 MCP 的智能旅行助手项目。

当前项目已经不是最初的“学习主题路由图”，而是一个真正的 Agent 原型：

- 使用 `MessagesState` 维护多轮对话
- 使用 `qwen3-max` 作为决策模型
- 使用高德 MCP 工具提供天气、地理编码、逆地理编码和地点提示能力
- 在进入模型前增加参数澄清与地点消歧层
- 使用 `checkpointer + thread_id` 恢复多轮上下文
- 使用 LangGraph 的 `assistant -> tools -> assistant` 循环完成工具调用

## 1. 项目结构

```text
langgraph/
├─ src/
│  └─ langgraph_study/
│     ├─ __init__.py
│     ├─ app/
│     │  ├─ __init__.py
│     │  └─ cli.py
│     ├─ assistant/
│     │  ├─ __init__.py
│     │  ├─ graph.py
│     │  ├─ nodes.py
│     │  └─ state.py
│     ├─ core/
│     │  ├─ __init__.py
│     │  └─ config.py
│     ├─ integrations/
│     │  ├─ __init__.py
│     │  ├─ llm.py
│     │  └─ mcp_tools.py
│     ├─ mcp/
│     │  ├─ __init__.py
│     │  └─ amap_server.py
├─ tests/
│  ├─ test_amap_mcp_server.py
│  ├─ test_graph.py
│  └─ test_main.py
├─ langgraph.json
├─ pyproject.toml
├─ README.md
└─ TODO.md
```

## 2. 当前工作流程

主图是一个真正的 Agent loop：

1. 用户输入旅行问题
2. `analyze_query` 节点先抽取意图、地点和时间线索
3. 如果地点不清晰，`clarify` 节点先向用户追问
4. 如果信息足够，`assistant` 节点调用 `qwen3-max`
5. 模型判断需要工具时，调用高德 MCP 工具
6. 工具结果回到 `assistant`
7. `checkpointer` 按 `thread_id` 持久化状态
8. 模型整理结果并输出最终回答

这比之前的“固定路由图”更接近真实 Agent。

## 3. 环境准备

### 3.1 创建 Conda 环境

```powershell
conda env create -f environment.yml
conda activate langgraph
```

### 3.2 安装依赖

```powershell
python -m pip install -U pip
python -m pip install -e .[dev]
```

### 3.3 配置模型与高德 Key

```powershell
conda env config vars set DASHSCOPE_API_KEY="你的 DashScope Key" QWEN_MODEL="qwen3-max" AMAP_API_KEY="你的高德 Key" -n langgraph
conda deactivate
conda activate langgraph
```

说明：

- 默认模型名已经是 `qwen3-max`
- 如果不设置 `QWEN_MODEL`，代码也会默认使用 `qwen3-max`
- 高德 MCP 默认读取 `AMAP_API_KEY`

## 4. 运行方式

### 4.1 启动旅行助手

```powershell
python -m langgraph_study.app.cli
```

启动后可以多轮提问，例如：

- `北京今天天气怎么样？`
- `上海和杭州这周末哪个更适合出行？`
- `帮我把天安门广场转成坐标`

如果只想单次执行：

```powershell
python -m langgraph_study.app.cli --input "北京今天天气怎么样？"
```

如果你希望在不同命令之间继续同一段对话，显式传同一个 `thread_id`：

```powershell
python -m langgraph_study.app.cli --thread-id trip-demo --input "北京今天天气怎么样？"
python -m langgraph_study.app.cli --thread-id trip-demo --input "那上海呢？"
```

说明：

- 当前 CLI 会打印正在使用的 `thread_id`
- `/reset` 会创建新的 `thread_id`
- checkpoint 默认写入 `.langgraph_data/checkpoints.sqlite`

### 4.2 查看图结构

```powershell
python -m langgraph_study.app.cli --show-mermaid --no-prompt
```

### 4.3 使用 LangGraph Studio

```powershell
langgraph dev
```

Studio 中可以直接提交：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "北京今天天气怎么样？"
    }
  ]
}
```

### 4.4 单独运行高德 MCP 服务

如果你想把高德工具单独暴露给别的客户端：

```powershell
python -m langgraph_study.mcp.amap_server --transport streamable-http
```

当前提供的 MCP 工具：

- `geocode`
- `reverse_geocode`
- `weather`
- `input_tips`

## 5. 代码入口说明

### `src/langgraph_study/assistant/graph.py`

定义主图结构与 checkpoint 接入：

- `analyze_query` 节点
- `clarify` 节点
- `assistant` 节点
- `tools` 节点
- `analyze_query -> clarify/assistant -> tools -> assistant` 流程
- `build_persistent_graph()` 为 CLI 构建持久化图

### `src/langgraph_study/assistant/nodes.py`

定义模型节点逻辑。这里不是手写业务路由，而是把系统提示词和对话消息送给模型。

### `src/langgraph_study/integrations/llm.py`

负责创建 `qwen3-max` 模型实例。

### `src/langgraph_study/integrations/mcp_tools.py`

负责通过 MCP client 加载高德工具，并接入 Agent。

### `src/langgraph_study/mcp/amap_server.py`

提供高德 MCP 服务本体。

### `src/langgraph_study/app/cli.py`

提供终端聊天入口，支持 `thread_id` 和多轮上下文恢复。

## 6. 学习顺序建议

当前阶段建议按这个顺序看：

1. 先看 `assistant/graph.py`，理解 Agent loop 与 checkpoint
2. 再看 `assistant/nodes.py`，理解澄清层和模型节点
3. 再看 `integrations/mcp_tools.py`，理解 MCP 工具如何接入 LangGraph
4. 再看 `mcp/amap_server.py`，理解工具服务本身如何实现
5. 最后运行 `app/cli.py` 和 Studio，看真实交互效果

## 7. 参考依据

本项目主要参考以下官方资料：

- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangChain MCP: https://docs.langchain.com/oss/python/langchain/mcp
- MCP Python SDK: https://py.sdk.modelcontextprotocol.io/
- LangChain Tongyi/Qwen 集成: https://docs.langchain.com/oss/python/integrations/chat/tongyi
