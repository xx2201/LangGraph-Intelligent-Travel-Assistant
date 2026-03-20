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
├─ docs/
│  └─ summaries/
│     ├─ 2026-03-20-init.md
│     └─ SUMMARY_TEMPLATE.md
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

这个最小版本包含一个“学习主题路由图”：

1. 接收你的学习主题
2. 分析主题属于哪一类
3. 路由到不同节点
4. 返回解释、学习重点和下一步建议

虽然它不调用 LLM，但已经完整覆盖了 LangGraph 最基础的图式编排思路。

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
python -m langgraph_study.main --topic "我想先理解 LangGraph 的 state"
```

### 3.4 查看图结构

```powershell
python -m langgraph_study.main --topic "条件边怎么工作" --show-mermaid
```

### 3.5 运行测试

```powershell
pytest
```

## 4. 代码入口说明

### `src/langgraph_study/state.py`

定义图的状态结构。你会看到：

- 哪些字段属于共享状态
- 哪些字段会在节点间持续流动
- `Annotated[..., operator.add]` 如何实现增量累积

### `src/langgraph_study/nodes.py`

定义节点逻辑。每个节点本质上都是：

- 读入当前状态
- 返回新的部分状态更新

这是 LangGraph 最重要的抽象之一。

### `src/langgraph_study/graph.py`

定义图结构，包括：

- `START`
- 普通边
- 条件边
- `compile()`

### `src/langgraph_study/main.py`

提供一个最小 CLI，便于你直接运行和观察结果。

## 5. 建议学习顺序

建议你按下面顺序理解，而不是直接抄更复杂 agent 示例：

1. 先看 `state.py`，理解状态为什么是图的中心
2. 再看 `nodes.py`，理解节点如何只返回“状态增量”
3. 再看 `graph.py`，理解控制流如何定义
4. 最后运行 `main.py`，观察输入如何沿图流动

这是比“先看 agent demo”更稳的路径。

## 6. 为什么这个项目故意不先接 LangChain Agent

因为你已经学过 LangChain，下一步真正需要补的是：

- LangGraph 到底解决什么问题
- 它与 LangChain 抽象层的边界在哪里
- 图编排和普通链式调用有何本质差异

如果一开始直接上“带工具调用的 agent”，容易只是在重复 LangChain 经验，而不是理解 LangGraph。

## 7. 下一阶段推荐路线

后续可以按这个顺序逐步演进：

1. 当前最小图：状态流转与条件路由
2. 加入 `MessagesState` 与消息流
3. 加入真实模型节点
4. 加入工具调用
5. 加入 `checkpointer` / memory
6. 加入子图 `subgraph`
7. 加入 human-in-the-loop
8. 加入 LangSmith tracing

## 8. 每次迭代的中文总结规则

从这次开始，后续你每次提出新需求，我都会额外产出一份中文总结，放到：

`docs/summaries/`

你可以直接把它作为 Git 提交说明的素材，或者整理成变更记录。

## 9. 参考依据

本项目结构和最小 API 参考的是 LangChain 官方 LangGraph 文档中的 v1.x 资料，尤其是以下方向：

- Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Quickstart: https://docs.langchain.com/oss/python/langgraph/quickstart
- Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api

注意：当前工程为了教学稳定性，没有照搬官方的“模型 + tools” quickstart，而是先抽出更基础的图编排骨架。

