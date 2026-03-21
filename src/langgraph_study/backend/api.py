from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

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


class ThreadStateView(BaseModel):
    """Small state summary shown in the thread detail panel."""

    query_context: dict
    message_count: int
    user_message_count: int
    assistant_message_count: int
    tool_call_count: int
    next_route_hint: str


class ThreadDetail(BaseModel):
    """Thread metadata together with restored message history and state summary."""

    thread: ThreadSummary
    messages: list[MessageView]
    state: ThreadStateView


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


def build_state_view(snapshot_values: dict) -> ThreadStateView:
    """Summarize the current LangGraph state for the frontend debug panel."""

    messages = snapshot_values.get("messages", [])
    query_context = snapshot_values.get("query_context", {})
    user_message_count = sum(isinstance(message, HumanMessage) for message in messages)
    assistant_message_count = sum(
        isinstance(message, AIMessage) and bool(getattr(message, "content", ""))
        for message in messages
    )
    tool_call_count = sum(
        len(message.tool_calls)
        for message in messages
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
    )
    next_route_hint = "clarify" if query_context.get("needs_clarification") else "assistant"
    return ThreadStateView(
        query_context=query_context,
        message_count=len(messages),
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        tool_call_count=tool_call_count,
        next_route_hint=next_route_hint,
    )


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
        state=build_state_view(values),
    )


def ndjson_line(payload: dict) -> bytes:
    """Encode one streaming event as newline-delimited JSON."""

    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


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
        """Run one turn and stream the final assistant reply to the frontend in chunks.

        This is UI-level streaming. The graph still completes one full turn internally,
        then the final reply is emitted chunk by chunk so the user no longer sees a
        blank screen until the entire answer is ready.
        """

        async def event_stream() -> AsyncIterator[bytes]:
            thread_id = request.thread_id or next_thread_id()
            await ensure_thread(app.state.thread_store, thread_id)
            yield ndjson_line({"type": "thread", "thread_id": thread_id})
            yield ndjson_line({"type": "assistant_start"})

            result = await app.state.graph.ainvoke(
                {"messages": [HumanMessage(content=request.message)]},
                config=make_graph_config(thread_id),
            )
            reply = extract_last_ai_text(result["messages"])
            record = await update_thread_after_chat(
                app.state.thread_store,
                thread_id=thread_id,
                user_message=request.message,
                assistant_message=reply,
            )
            detail = await load_thread_detail_data(app.state.graph, record)

            for chunk in chunk_text(reply):
                yield ndjson_line({"type": "assistant_delta", "content": chunk})
                await asyncio.sleep(0.02)

            yield ndjson_line(
                {
                    "type": "done",
                    "thread": detail.thread.model_dump(),
                    "state": detail.state.model_dump(),
                    "messages": [message.model_dump() for message in detail.messages],
                }
            )

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return app
