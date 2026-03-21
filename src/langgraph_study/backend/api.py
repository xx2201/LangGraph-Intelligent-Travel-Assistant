from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..assistant.graph import build_persistent_graph
from ..app.cli import extract_last_ai_text, make_graph_config, next_thread_id


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    """Request body for the browser chat API."""

    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Response body returned to the browser client."""

    thread_id: str
    reply: str


def get_frontend_dir() -> Path:
    """Return the directory that stores static frontend files."""

    return FRONTEND_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and clean up the persistent LangGraph runtime for web requests."""

    app.state.graph = await build_persistent_graph()
    try:
        yield
    finally:
        checkpointer = getattr(app.state.graph, "checkpointer", None)
        connection = getattr(checkpointer, "conn", None)
        if connection is not None:
            await connection.close()


def create_app() -> FastAPI:
    """Create the FastAPI application used by the browser frontend."""

    app = FastAPI(title="LangGraph Travel Assistant API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=get_frontend_dir()), name="static")

    @app.get("/", response_class=FileResponse)
    async def home() -> FileResponse:
        """Serve the static HTML entry page."""

        return FileResponse(get_frontend_dir() / "index.html")

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Run one turn of the LangGraph agent and return the latest assistant reply."""

        thread_id = request.thread_id or next_thread_id()
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=make_graph_config(thread_id),
        )
        return ChatResponse(
            thread_id=thread_id,
            reply=extract_last_ai_text(result["messages"]),
        )

    return app
