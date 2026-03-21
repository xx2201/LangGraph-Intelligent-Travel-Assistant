from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
    """Request body for one chat turn from the browser frontend."""

    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Response body returned after one chat turn completes."""

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
    """Convert LangGraph messages into a frontend-friendly form.

    Tool messages and empty assistant tool-call stubs are filtered out because the
    frontend thread window is meant to show the user-facing conversation only.
    """

    if isinstance(message, HumanMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        return MessageView(role="user", content=content)

    if isinstance(message, AIMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        if not content:
            return None
        return MessageView(role="assistant", content=content)

    return None


async def load_thread_messages(graph, thread_id: str) -> list[MessageView]:
    """Read one thread's current message history from the LangGraph checkpoint store."""

    snapshot = await graph.aget_state(make_graph_config(thread_id))
    messages = snapshot.values.get("messages", [])
    serialized: list[MessageView] = []
    for message in messages:
        converted = message_to_view(message)
        if converted is not None:
            serialized.append(converted)
    return serialized


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
        """Return metadata and restored message history for a specific thread."""

        record = await get_thread(app.state.thread_store, thread_id)
        if record is None:
            raise HTTPException(status_code=404, detail="thread not found")
        messages = await load_thread_messages(app.state.graph, thread_id)
        return ThreadDetail(thread=summarize_thread_record(record), messages=messages)

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

    return app
