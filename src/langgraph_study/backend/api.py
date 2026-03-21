from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from ..assistant.graph import build_persistent_graph
from ..app.cli import extract_last_ai_text, make_graph_config, next_thread_id
from ..core.config import THREAD_STORE_DB_PATH
from .thread_store import (
    connect_thread_store,
    create_thread,
    ensure_thread,
    get_thread,
    initialize_thread_store,
    list_threads,
    update_thread_after_chat,
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    """Request body for one full chat turn."""

    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Response body returned after one full chat turn completes."""

    thread_id: str
    reply: str


class ThreadSummary(BaseModel):
    """Compact thread metadata used by the left-side session list."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_user_message: str
    last_assistant_message: str


class MessageView(BaseModel):
    """Simplified user-facing message shape returned to the frontend."""

    role: str
    content: str


class ThreadDetail(BaseModel):
    """Thread metadata together with restored message history."""

    thread: ThreadSummary
    messages: list[MessageView]


def get_frontend_dir() -> Path:
    """Return the directory that stores static frontend files."""

    return FRONTEND_DIR


def summarize_thread_record(record) -> ThreadSummary:
    """Convert a backend thread record into the response model used by the API."""

    return ThreadSummary(**record.to_dict())


def message_to_view(message: BaseMessage) -> MessageView | None:
    """Convert LangGraph messages into a frontend-friendly form."""

    if isinstance(message, HumanMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        return MessageView(role="user", content=content)

    if isinstance(message, AIMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        if not content:
            return None
        return MessageView(role="assistant", content=content)

    return None


def chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    """Split the final assistant reply into small chunks for UI streaming."""

    if not text:
        return [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


async def load_thread_detail_data(graph, thread_record) -> ThreadDetail:
    """Read one thread's state snapshot and convert it to a frontend-friendly payload."""

    snapshot = await graph.aget_state(make_graph_config(thread_record.thread_id))
    values = snapshot.values
    serialized_messages: list[MessageView] = []
    for message in values.get("messages", []):
        converted = message_to_view(message)
        if converted is not None:
            serialized_messages.append(converted)
    return ThreadDetail(
        thread=summarize_thread_record(thread_record),
        messages=serialized_messages,
    )


def ndjson_line(payload: dict) -> bytes:
    """Encode one streaming event as newline-delimited JSON."""

    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def extract_stream_text(chunk) -> str:
    """Extract user-facing text from a streamed model chunk event."""

    if chunk is None:
        return ""

    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return str(content) if content else ""


def summarize_value(value: Any, limit: int = 96) -> str:
    """Convert nested event payloads into short human-readable timeline text."""

    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 1]}..."


def summarize_query_context(query_context: dict[str, Any]) -> str:
    """Turn structured query context into a compact one-line explanation."""

    parts: list[str] = []
    intent = query_context.get("intent")
    if intent:
        parts.append(f"意图: {intent}")
    city = query_context.get("normalized_city") or query_context.get("location_text")
    if city:
        parts.append(f"地点: {city}")
    time_text = query_context.get("time_text")
    if time_text:
        parts.append(f"时间: {time_text}")
    if query_context.get("needs_clarification"):
        reason = query_context.get("clarification_reason") or "信息不足"
        parts.append(f"需澄清: {reason}")
    return " | ".join(parts) if parts else "已完成查询预分析。"


def build_process_updates(event: dict[str, Any], tracker: dict[str, Any]) -> list[dict[str, Any]]:
    """Map LangGraph runtime events to a compact UI timeline model."""

    event_type = event.get("event")
    name = event.get("name")
    data = event.get("data", {})
    run_id = event.get("run_id", "")
    updates: list[dict[str, Any]] = []

    if event_type == "on_chain_start" and name == "analyze_query":
        updates.append(
            {
                "type": "process",
                "key": "analyze_query",
                "stage": "analysis",
                "status": "running",
                "title": "分析用户输入",
                "detail": "提取意图、地点和时间线索。",
            }
        )
    elif event_type == "on_chain_end" and name == "analyze_query":
        query_context = data.get("output", {}).get("query_context", {})
        updates.append(
            {
                "type": "process",
                "key": "analyze_query",
                "stage": "analysis",
                "status": "done",
                "title": "分析完成",
                "detail": summarize_query_context(query_context),
            }
        )
    elif event_type == "on_chain_end" and name == "route_after_analysis":
        route = data.get("output", "")
        detail = "信息不足，先进入澄清节点。" if route == "clarify" else "信息足够，进入助手节点。"
        updates.append(
            {
                "type": "process",
                "key": "route_after_analysis",
                "stage": "route",
                "status": "done",
                "title": "选择下一步",
                "detail": detail,
            }
        )
    elif event_type == "on_chain_start" and name == "clarify":
        updates.append(
            {
                "type": "process",
                "key": "clarify",
                "stage": "clarify",
                "status": "running",
                "title": "请求澄清",
                "detail": "当前信息不足，正在生成追问。",
            }
        )
    elif event_type == "on_chain_end" and name == "clarify":
        updates.append(
            {
                "type": "process",
                "key": "clarify",
                "stage": "clarify",
                "status": "done",
                "title": "澄清完成",
                "detail": "已生成追问，等待用户补充信息。",
            }
        )
    elif event_type == "on_chat_model_start":
        tracker["assistant_round"] += 1
        assistant_key = f"assistant_{tracker['assistant_round']}"
        tracker["current_assistant_key"] = assistant_key
        updates.append(
            {
                "type": "process",
                "key": assistant_key,
                "stage": "assistant",
                "status": "running",
                "title": f"助手回合 {tracker['assistant_round']}",
                "detail": "模型正在生成下一步决策或回答。",
            }
        )
    elif event_type == "on_chat_model_end":
        assistant_key = tracker.get("current_assistant_key", "assistant_1")
        updates.append(
            {
                "type": "process",
                "key": assistant_key,
                "stage": "assistant",
                "status": "done",
                "title": f"助手回合 {tracker.get('assistant_round', 1)}",
                "detail": "模型本轮输出完成，可能包含工具调用或最终回答。",
            }
        )
    elif event_type == "on_tool_start":
        tracker["tool_round"] += 1
        tool_key = f"tool_{tracker['tool_round']}"
        tracker["tool_runs"][run_id] = tool_key
        tool_input = summarize_value(data.get("input"))
        detail = f"输入: {tool_input}" if tool_input else "正在调用外部工具。"
        updates.append(
            {
                "type": "process",
                "key": tool_key,
                "stage": "tool",
                "status": "running",
                "title": f"调用工具: {name}",
                "detail": detail,
            }
        )
    elif event_type == "on_tool_end":
        tool_key = tracker["tool_runs"].pop(run_id, f"tool_{tracker['tool_round']}")
        tool_output = summarize_value(data.get("output"))
        detail = f"输出: {tool_output}" if tool_output else "工具调用完成。"
        updates.append(
            {
                "type": "process",
                "key": tool_key,
                "stage": "tool",
                "status": "done",
                "title": f"工具完成: {name}",
                "detail": detail,
            }
        )
    elif event_type == "on_chain_end" and name == "LangGraph":
        updates.append(
            {
                "type": "process",
                "key": "graph_complete",
                "stage": "graph",
                "status": "done",
                "title": "本轮完成",
                "detail": "Agent 已完成本轮图执行。",
            }
        )

    return updates


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and clean up the persistent graph and thread metadata store."""

    app.state.graph = await build_persistent_graph()
    app.state.thread_store = await connect_thread_store(THREAD_STORE_DB_PATH)
    await initialize_thread_store(app.state.thread_store)
    try:
        yield
    finally:
        checkpointer = getattr(app.state.graph, "checkpointer", None)
        connection = getattr(checkpointer, "conn", None)
        if connection is not None:
            await connection.close()
        await app.state.thread_store.close()


def create_app() -> FastAPI:
    """Create the FastAPI application used by the browser frontend."""

    app = FastAPI(title="LangGraph Travel Assistant API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=get_frontend_dir()), name="static")

    @app.get("/", response_class=FileResponse)
    async def home() -> FileResponse:
        """Serve the static HTML entry page."""

        return FileResponse(get_frontend_dir() / "index.html")

    @app.get("/api/threads", response_model=list[ThreadSummary])
    async def get_threads() -> list[ThreadSummary]:
        """Return all backend-managed chat threads."""

        records = await list_threads(app.state.thread_store)
        return [summarize_thread_record(record) for record in records]

    @app.post("/api/threads", response_model=ThreadSummary)
    async def post_thread() -> ThreadSummary:
        """Create a new empty chat thread and return its metadata."""

        thread_id = next_thread_id()
        record = await create_thread(app.state.thread_store, thread_id)
        return summarize_thread_record(record)

    @app.get("/api/threads/{thread_id}", response_model=ThreadDetail)
    async def get_thread_detail(thread_id: str) -> ThreadDetail:
        """Return metadata, restored messages, and state visualization for a thread."""

        record = await get_thread(app.state.thread_store, thread_id)
        if record is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return await load_thread_detail_data(app.state.graph, record)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Run one turn of the LangGraph agent and update thread metadata."""

        thread_id = request.thread_id or next_thread_id()
        await ensure_thread(app.state.thread_store, thread_id)
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=make_graph_config(thread_id),
        )
        reply = extract_last_ai_text(result["messages"])
        await update_thread_after_chat(
            app.state.thread_store,
            thread_id=thread_id,
            user_message=request.message,
            assistant_message=reply,
        )
        return ChatResponse(thread_id=thread_id, reply=reply)

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Run one turn and stream assistant output to the frontend.

        The implementation first tries to forward real model stream events coming from
        ``graph.astream_events()``. This preserves the ``assistant -> tools ->
        assistant`` control flow: tool-call-only phases emit no text, while the final
        assistant phase can stream visible content. If the underlying model/tool stack
        does not yield visible stream chunks, the endpoint falls back to chunking the
        final answer so the UI still behaves incrementally.
        """

        async def event_stream() -> AsyncIterator[bytes]:
            thread_id = request.thread_id or next_thread_id()
            await ensure_thread(app.state.thread_store, thread_id)
            yield ndjson_line({"type": "thread", "thread_id": thread_id})
            yielded_text = False
            started_assistant = False
            process_tracker = {
                "assistant_round": 0,
                "current_assistant_key": "",
                "tool_round": 0,
                "tool_runs": {},
            }
            yield ndjson_line(
                {
                    "type": "process",
                    "key": "graph_start",
                    "stage": "graph",
                    "status": "running",
                    "title": "收到新请求",
                    "detail": "LangGraph 正在启动本轮执行。",
                }
            )

            async for event in app.state.graph.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                config=make_graph_config(thread_id),
                version="v2",
            ):
                for process_update in build_process_updates(event, process_tracker):
                    yield ndjson_line(process_update)

                if event.get("event") != "on_chat_model_stream":
                    continue
                chunk = event.get("data", {}).get("chunk")
                text = extract_stream_text(chunk)
                if not text:
                    continue
                if not started_assistant:
                    yield ndjson_line({"type": "assistant_start"})
                    started_assistant = True
                yielded_text = True
                yield ndjson_line({"type": "assistant_delta", "content": text})

            detail_before_update = await load_thread_detail_data(
                app.state.graph,
                await ensure_thread(app.state.thread_store, thread_id),
            )
            reply = ""
            for message in reversed(detail_before_update.messages):
                if message.role == "assistant":
                    reply = message.content
                    break

            record = await update_thread_after_chat(
                app.state.thread_store,
                thread_id=thread_id,
                user_message=request.message,
                assistant_message=reply,
            )
            detail = await load_thread_detail_data(app.state.graph, record)

            if not started_assistant:
                yield ndjson_line({"type": "assistant_start"})
                started_assistant = True

            if not yielded_text:
                for chunk in chunk_text(reply):
                    yield ndjson_line({"type": "assistant_delta", "content": chunk})
                    await asyncio.sleep(0.02)

            yield ndjson_line(
                {
                    "type": "done",
                    "thread": detail.thread.model_dump(),
                    "messages": [message.model_dump() for message in detail.messages],
                }
            )

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return app
