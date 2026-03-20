# LangGraph 学习工程

这是一个用于系统学习 LangGraph 的最小项目骨架。

你的前置背景是已经学过 LangChain，所以这个工程刻意不从“如何调用模型”开始，而是先把 LangGraph 自己最核心的对象和运行方式拆开：

- `State`
- `Node`
- `Edge`
- `Conditional Edge`
- `compile()`
- 图执行结果的可观察性

第一阶段先使用**无模型依赖**的确定性节点，目的是把“图编排”理解清楚。  
如果一开始就接 LLM，很多现象会被模型输出噪声掩盖，不利于建立稳定认知。

## 1. 项目结构

```text
langgraph/
├─ src/
│  └─ langgraph_study/
│     ├─ __init__.py
│     ├─ graph.py
│     ├─ main.py
│     ├─ nodes.py
│     └─ state.py
├─ tests/
│  └─ test_graph.py
├─ .gitignore
├─ pyproject.toml
├─ README.md
└─ TODO.md
```

## 2. 你现在能学到什么

这个版本包含一个“学习主题路由图”，并且已经支持接入真实模型 `qwen3-max`：

1. 接收你的自然语言学习输入
2. 分析主题属于哪一类
3. 路由到不同节点
4. 优先调用 `qwen3-max` 生成解释
5. 返回解释、学习重点和下一步建议

如果没有配置 `DASHSCOPE_API_KEY`，项目会自动回退到本地教学答案，因此仍然可以学习图编排逻辑。

## 3. 快速开始

### 3.1 创建 Conda 环境

```powershell
conda env create -f environment.yml
conda activate langgraph
```

如果你只想手动创建，也可以：

```powershell
conda create -n langgraph python=3.11 pip -y
conda activate langgraph
python -m pip install -U pip
python -m pip install -e .[dev]
```

### 3.2 安装依赖

如果你使用的是 `environment.yml`，环境创建时会自动安装依赖。  
如果你是手动建环境，再执行上面的 `pip install -e .[dev]` 即可。

### 3.3 运行示例

```powershell
python -m langgraph_study.main
```

启动后，程序会在终端中提示你输入学习主题和背景信息。

如果你仍然希望用命令参数传入，也可以：

```powershell
python -m langgraph_study.main --input "我想先理解 LangGraph 的 state" --background "已学习过 LangChain"
```

### 3.4 查看图结构

```powershell
python -m langgraph_study.main --show-mermaid
```

### 3.5 配置 `qwen3-max`

当前项目通过 LangChain 官方 Qwen 集成访问真实模型。按官方文档，需要设置 `DASHSCOPE_API_KEY` 环境变量。

如果你使用 Conda：

```powershell
conda env config vars set DASHSCOPE_API_KEY="你的 DashScope Key" QWEN_MODEL="qwen3-max" -n langgraph
conda deactivate
conda activate langgraph
```

说明：

- 默认模型名已经是 `qwen3-max`
- 如果不设置 `QWEN_MODEL`，代码也会默认使用 `qwen3-max`
- 如果没有 `DASHSCOPE_API_KEY`，程序会回退到本地答案，不会直接崩溃

### 3.6 运行测试

```powershell
pytest
```

### 3.7 使用 LangSmith 观察执行轨迹

当前项目已经支持通过环境变量把图执行轨迹发送到 LangSmith。

如果你使用的是我已经配置好的本地 `conda` 环境，只需要重新激活一次环境：

```powershell
conda deactivate
conda activate langgraph
```

然后直接运行：

```powershell
python -m langgraph_study.main
```

执行完成后，可以到 LangSmith 中查看这次图执行的 trace。

说明：

- 当前项目没有把 API Key 写进仓库文件
- LangSmith 配置保存在你的本地 `conda` 环境变量中
- 这比把密钥写入代码或 `README` 更安全

### 3.8 使用 LangGraph Studio

当前项目已经安装了 LangGraph CLI，并补充了 [langgraph.json](g:\mycode\langgraph\langgraph.json)，可以直接用于本地 Studio 开发模式。

先激活环境：

```powershell
conda activate langgraph
```

然后启动开发服务器：

```powershell
langgraph dev
```

如果你的终端里 `langgraph` 不在 `PATH`，可以直接运行：

```powershell
G:\ANACONDA\envs\langgraph\Scripts\langgraph.exe dev
```

说明：

- `langgraph dev` 不需要 Docker
- 它会读取项目根目录下的 `langgraph.json`
- 当前图入口是 `study_graph -> langgraph_study.graph:graph`
- 启动后会连接 Studio，用来查看图、状态和运行过程

Studio 中建议直接提交：

```json
{
  "input": "我想先理解 LangGraph 的 state"
}
```

如果你想补充背景，也可以：

```json
{
  "input": "我想先理解 LangGraph 的 state",
  "background": "已学习过 LangChain"
}
```

## 4. 代码入口说明

### `src/langgraph_study/state.py`

定义图的状态结构。你会看到：

- `input` 如何承载更自然的 Studio 输入
- 哪些字段属于共享状态
- 哪些字段会在节点间持续流动
- `Annotated[..., operator.add]` 如何实现增量累积

### `src/langgraph_study/nodes.py`

定义节点逻辑。每个节点本质上都是：

- 读入当前状态
- 返回新的部分状态更新

这是 LangGraph 最重要的抽象之一。

### `src/langgraph_study/llm.py`

封装真实模型调用与回退逻辑。你会看到：

- 如何接入 `qwen3-max`
- 如何通过环境变量启用真实模型
- 没有 API Key 时如何回退到本地教学答案

### `src/langgraph_study/graph.py`

定义图结构，包括：

- `START`
- 普通边
- 条件边
- `compile()`

### `src/langgraph_study/main.py`

提供一个最小 CLI，便于你直接运行、交互式输入和观察结果。

## 5. 建议学习顺序

建议你按下面顺序理解，而不是直接抄更复杂 agent 示例：

1. 先看 `state.py`，理解状态为什么是图的中心
2. 再看 `nodes.py`，理解节点如何只返回“状态增量”
3. 再看 `graph.py`，理解控制流如何定义
4. 再看 `llm.py`，理解真实模型是如何被接入图中的
5. 最后运行 `main.py`，观察输入如何沿图流动

这是比“先看 agent demo”更稳的路径。

## 6. 为什么这个项目故意不先接 LangChain Agent

因为你已经学过 LangChain，下一步真正需要补的是：

- LangGraph 到底解决什么问题
- 它与 LangChain 抽象层的边界在哪里
- 图编排和普通链式调用有何本质差异

如果一开始直接上“带工具调用的 agent”，容易只是在重复 LangChain 经验，而不是理解 LangGraph。

## 7. 下一阶段推荐路线

后续可以按这个顺序逐步演进：

1. 当前阶段：自然输入 + 路由 + `qwen3-max`
2. 加入 `MessagesState` 与消息流
3. 加入工具调用
4. 加入 `checkpointer` / memory
5. 加入子图 `subgraph`
6. 加入 human-in-the-loop
7. 深入 LangSmith tracing

## 8. 参考依据

本项目结构和最小 API 参考的是 LangChain 官方 LangGraph 文档中的 v1.x 资料，尤其是以下方向：

- Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Quickstart: https://docs.langchain.com/oss/python/langgraph/quickstart
- Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api

注意：当前工程为了教学稳定性，没有照搬官方的“模型 + tools” quickstart，而是先抽出更基础的图编排骨架。
