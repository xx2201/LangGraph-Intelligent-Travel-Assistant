# `graph.ainvoke(...)` 逐帧状态变化

下面按“新 `thread_id`、运行时图、模型遵守系统提示并选择调天气工具”这个前提来讲。
也就是最典型路径：

```python
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="北京天气怎么样？")]},
    config={"configurable": {"thread_id": "thread-1"}}
)
```

图结构定义在 [src/langgraph_study/assistant/graph.py](/g:/mycode/langgraph/src/langgraph_study/assistant/graph.py#L27)，节点逻辑在 [src/langgraph_study/assistant/nodes.py](/g:/mycode/langgraph/src/langgraph_study/assistant/nodes.py#L79)，状态类型在 [src/langgraph_study/assistant/state.py](/g:/mycode/langgraph/src/langgraph_study/assistant/state.py#L23)。

## 先说一个关键点

`state` 里真正持久化的是：
- `messages`
- `query_context`

系统提示词和“上下文提示词”不会写回 state，它们只是 `assistant` 节点在调用模型时临时拼出来的。

## 0. 进入图之前

输入给 `ainvoke()` 的初始 state 是：

```python
{
    "messages": [
        HumanMessage(content="北京天气怎么样？")
    ]
}
```

如果这个 `thread_id` 之前已有 checkpoint，那么 LangGraph 会先恢复旧 state，再把这次输入合进去。
这里按“全新会话”讲，所以初始 state 就这一条用户消息。

## 1. `START -> analyze_query`

`START` 是虚拟起点，不改 state。
根据 [src/langgraph_study/assistant/graph.py](/g:/mycode/langgraph/src/langgraph_study/assistant/graph.py#L46)，第一跳固定进入 `analyze_query`。

进入 `analyze_query` 前的 state：

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？")
    ]
}
```

`analyze_query(state)` 做的事见 [src/langgraph_study/assistant/nodes.py](/g:/mycode/langgraph/src/langgraph_study/assistant/nodes.py#L151)：

1. 调 `extract_latest_user_text(state)`
   从 `messages` 里倒序找到最后一个 `HumanMessage`
   得到：
   ```python
   "北京天气怎么样？"
   ```

2. 调 `build_query_context(user_text)`
   这一步又继续拆成几步：

   - `text = user_text.strip()`
     得到 `"北京天气怎么样？"`

   - `detect_intent(text)`
     因为命中“天气”关键词，返回：
     ```python
     "weather"
     ```

   - `detect_time_text(text)`
     没有命中“今天 / 明天 / 周末”等词，返回：
     ```python
     ""
     ```

   - `extract_location_text(text, intent="weather")`
     用天气相关正则抽地点片段，能提取出：
     ```python
     "北京"
     ```

   - `normalize_city("北京")`
     命中城市别名表，返回：
     ```python
     "北京"
     ```

   - `assess_clarification_need(...)`
     因为：
     - 意图是 `weather`
     - 有地点 `"北京"`
     - 且 `normalized_city` 已识别成功

     所以返回：
     ```python
     (False, "")
     ```

   最终生成的 `query_context` 是：

   ```python
   {
       "raw_user_input": "北京天气怎么样？",
       "intent": "weather",
       "time_text": "",
       "needs_clarification": False,
       "clarification_reason": "",
       "location_text": "北京",
       "normalized_city": "北京"
   }
   ```

3. `analyze_query` 返回：
   ```python
   {"query_context": context}
   ```

LangGraph 把这个返回值合并进 state。
所以离开 `analyze_query` 后的 state 变成：

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？")
    ],
    "query_context": {
        "raw_user_input": "北京天气怎么样？",
        "intent": "weather",
        "time_text": "",
        "needs_clarification": False,
        "clarification_reason": "",
        "location_text": "北京",
        "normalized_city": "北京"
    }
}
```

## 2. `analyze_query -> route_after_analysis`

这是条件路由，不改 state，只决定下一跳。
根据 [src/langgraph_study/assistant/nodes.py](/g:/mycode/langgraph/src/langgraph_study/assistant/nodes.py#L159)，`route_after_analysis(state)` 会读：

```python
state["query_context"]["needs_clarification"]
```

当前是 `False`，所以返回：

```python
"assistant"
```

因此不会进入 `clarify`，而是进入 `assistant`。

此时 state 不变：

```python
{
    "messages": [HumanMessage("北京天气怎么样？")],
    "query_context": {...}
}
```

## 3. 第一次进入 `assistant`

`assistant` 节点是 `create_assistant_node(...)` 生成的 runnable，定义见 [src/langgraph_study/assistant/nodes.py](/g:/mycode/langgraph/src/langgraph_study/assistant/nodes.py#L79)。

这里会先调用内部的 `build_messages(state)`。它临时拼给模型的消息是：

1. `SystemMessage(TRAVEL_AGENT_SYSTEM_PROMPT)`
2. `SystemMessage(build_query_context_message(query_context))`
3. `HumanMessage("北京天气怎么样？")`

注意，这两条 `SystemMessage` 不会写回 state。

`build_query_context_message(query_context)` 大致会生成这样的文本：

```text
你会收到一个前置解析层生成的查询上下文。
如果上下文里已有标准城市名，天气查询优先使用该城市作为工具参数。
如果上下文提示地点不清晰，不要自行猜测城市。
最新用户问题：北京天气怎么样？
解析意图：weather
原始地点片段：北京
标准城市名：北京
```

然后 `assistant_async(state)` 调 `run_bound_model(bound_model, messages)`。

`run_bound_model(...)` 的执行顺序是：

1. 如果模型支持 `astream()`，优先流式消费
2. 否则如果支持 `ainvoke()`，走异步调用
3. 再否则退回 `invoke()`

因为这个模型之前已经 `bind_tools(resolved_tools)`，所以它这次不只是能回答文本，也可能产出 `tool_calls`。

对于这句“北京天气怎么样？”，在当前系统提示下，合理且预期的模型输出是一个带工具调用的 `AIMessage`，类似：

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "weather",
            "args": {"city": "北京"},
            "id": "call_weather_1",
            "type": "tool_call"
        }
    ]
)
```

于是 `assistant` 节点返回：

```python
{
    "messages": [AIMessage(...tool_calls...)]
}
```

由于 `messages` 是 `MessagesState` 里的聚合字段，LangGraph 不会覆盖旧消息，而是追加。
离开第一次 `assistant` 后，state 变成：

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather",
                    "args": {"city": "北京"},
                    "id": "call_weather_1",
                    "type": "tool_call"
                }
            ]
        )
    ],
    "query_context": {
        "raw_user_input": "北京天气怎么样？",
        "intent": "weather",
        "time_text": "",
        "needs_clarification": False,
        "clarification_reason": "",
        "location_text": "北京",
        "normalized_city": "北京"
    }
}
```

## 4. `assistant -> tools_condition`

这也是条件路由，不改 state。
`tools_condition(state)` 会检查最后一条 AIMessage 有没有 `tool_calls`。

当前最后一条消息确实有，所以返回：

```python
"tools"
```

于是进入工具节点。

state 不变。

## 5. 进入 `tools`

`tools` 节点是 `ToolNode(resolved_tools)`。
它会读取最后一条 AIMessage 中的 `tool_calls`，逐个执行。

当前只会执行一个调用：

```python
weather(city="北京")
```

如果这里接的是高德 MCP 工具，那么工具本身会去查真实天气，并返回结构化结果。
`ToolNode` 会把工具结果包装成 `ToolMessage` 加回消息列表。它的内容通常是工具输出的序列化结果，形态大致像这样：

```python
ToolMessage(
    content='{"query":{"city":"北京","extensions":"base"},"lives":[...],"status":"1","info":"OK"}',
    tool_call_id="call_weather_1"
)
```

所以离开 `tools` 后，state 变成：

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather",
                    "args": {"city": "北京"},
                    "id": "call_weather_1",
                    "type": "tool_call"
                }
            ]
        ),
        ToolMessage(
            content="...北京天气结果...",
            tool_call_id="call_weather_1"
        )
    ],
    "query_context": {
        "raw_user_input": "北京天气怎么样？",
        "intent": "weather",
        "time_text": "",
        "needs_clarification": False,
        "clarification_reason": "",
        "location_text": "北京",
        "normalized_city": "北京"
    }
}
```

## 6. `tools -> assistant`，再次进入 `assistant`

根据 [src/langgraph_study/assistant/graph.py](/g:/mycode/langgraph/src/langgraph_study/assistant/graph.py#L57)，`tools` 后固定回到 `assistant`。

第二次进入 `assistant` 时，`build_messages(state)` 拼给模型的内容会变成：

1. 系统提示词
2. 查询上下文提示词
3. `HumanMessage("北京天气怎么样？")`
4. 带 `tool_calls` 的 `AIMessage`
5. `ToolMessage("...北京天气结果...")`

也就是说，模型这次已经看到了工具结果。
因此它通常会输出一条最终回答，不再继续调用工具，例如：

```python
AIMessage(content="北京当前天气晴，气温 26 摄氏度，整体适合出门。")
```

于是 `assistant` 节点返回：

```python
{
    "messages": [
        AIMessage(content="北京当前天气晴，气温 26 摄氏度，整体适合出门。")
    ]
}
```

离开第二次 `assistant` 后，state 变成：

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="...北京天气结果...", tool_call_id="call_weather_1"),
        AIMessage(content="北京当前天气晴，气温 26 摄氏度，整体适合出门。")
    ],
    "query_context": {
        "raw_user_input": "北京天气怎么样？",
        "intent": "weather",
        "time_text": "",
        "needs_clarification": False,
        "clarification_reason": "",
        "location_text": "北京",
        "normalized_city": "北京"
    }
}
```

## 7. 第二次 `assistant -> tools_condition -> END`

`tools_condition(state)` 再检查最后一条 AIMessage。
这次它没有 `tool_calls`，所以返回：

```python
"__end__"
```

图执行结束，进入 `END`。

`END` 不改 state。
`ainvoke()` 最终返回的就是这一版完整 state。

## 最终返回值可以近似理解为

```python
{
    "messages": [
        HumanMessage("北京天气怎么样？"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="...北京天气结果...", tool_call_id="call_weather_1"),
        AIMessage(content="北京当前天气晴，气温 26 摄氏度，整体适合出门。")
    ],
    "query_context": {
        "raw_user_input": "北京天气怎么样？",
        "intent": "weather",
        "time_text": "",
        "needs_clarification": False,
        "clarification_reason": "",
        "location_text": "北京",
        "normalized_city": "北京"
    }
}
```

## 你要特别注意的三点

1. `query_context` 只在 `analyze_query` 写一次，后面节点只读它。
2. `assistant` 节点不会把系统提示词写回 `state["messages"]`，写回去的只有模型返回的 `AIMessage`。
3. `tools_condition` 和 `route_after_analysis` 都只是“路由函数”，不修改 state。
