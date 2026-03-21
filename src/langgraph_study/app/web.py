from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..assistant.graph import build_persistent_graph
from .cli import extract_last_ai_text, make_graph_config, next_thread_id


def render_homepage() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangGraph 旅行助手</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #f4f1ea; color: #1f2937; }
    .page { max-width: 900px; margin: 0 auto; padding: 32px 20px 48px; }
    .card { background: #fffdf8; border: 1px solid #e8dcc9; border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(78, 52, 24, 0.08); }
    h1 { margin-top: 0; font-size: 32px; }
    .muted { color: #6b7280; margin-bottom: 16px; }
    .toolbar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; }
    .thread { padding: 10px 12px; border-radius: 10px; background: #f3e8d4; font-size: 14px; }
    textarea { width: 100%; min-height: 120px; border: 1px solid #d6c4ab; border-radius: 12px; padding: 14px; font-size: 16px; resize: vertical; box-sizing: border-box; }
    button { border: 0; border-radius: 999px; padding: 12px 18px; cursor: pointer; font-size: 15px; }
    .primary { background: #a34718; color: white; }
    .secondary { background: #efe6d9; color: #5b4635; }
    .output { margin-top: 18px; white-space: pre-wrap; line-height: 1.6; background: #faf6ef; border: 1px solid #e8dcc9; border-radius: 12px; padding: 16px; min-height: 120px; }
    .hint { font-size: 13px; color: #6b7280; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>LangGraph 旅行助手</h1>
      <div class="muted">这是一个最小 FastAPI 前端。它调用的是同一个 LangGraph 运行时图，保留 thread_id 续聊能力。</div>
      <div class="toolbar">
        <div class="thread">当前 thread_id: <span id="threadId">未创建</span></div>
        <button class="secondary" id="newThreadBtn" type="button">新建会话</button>
      </div>
      <textarea id="messageInput" placeholder="例如：帮我做一个去成都的旅游规划"></textarea>
      <div class="toolbar">
        <button class="primary" id="sendBtn" type="button">发送</button>
      </div>
      <div class="output" id="outputBox">等待输入...</div>
      <div class="hint">输入会发送到 LangGraph 运行时图；同一个 thread_id 会自动续接上下文。</div>
    </div>
  </div>
  <script>
    const outputBox = document.getElementById("outputBox");
    const messageInput = document.getElementById("messageInput");
    const threadIdEl = document.getElementById("threadId");
    const sendBtn = document.getElementById("sendBtn");
    const newThreadBtn = document.getElementById("newThreadBtn");

    function getThreadId() {
      return localStorage.getItem("langgraph-thread-id") || "";
    }

    function setThreadId(threadId) {
      if (threadId) {
        localStorage.setItem("langgraph-thread-id", threadId);
        threadIdEl.textContent = threadId;
      } else {
        localStorage.removeItem("langgraph-thread-id");
        threadIdEl.textContent = "未创建";
      }
    }

    async function sendMessage() {
      const message = messageInput.value.trim();
      if (!message) {
        outputBox.textContent = "请输入问题。";
        return;
      }

      sendBtn.disabled = true;
      outputBox.textContent = "正在调用 LangGraph...";

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          thread_id: getThreadId()
        })
      });

      const data = await response.json();
      setThreadId(data.thread_id);
      outputBox.textContent = data.reply;
      sendBtn.disabled = false;
    }

    sendBtn.addEventListener("click", sendMessage);
    newThreadBtn.addEventListener("click", () => {
      setThreadId("");
      outputBox.textContent = "已创建新会话。";
    });

    setThreadId(getThreadId());
  </script>
</body>
</html>"""


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = await build_persistent_graph()
    try:
        yield
    finally:
        checkpointer = getattr(app.state.graph, "checkpointer", None)
        connection = getattr(checkpointer, "conn", None)
        if connection is not None:
            await connection.close()


app = FastAPI(title="LangGraph Travel Assistant UI", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return render_homepage()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or next_thread_id()
    result = await app.state.graph.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=make_graph_config(thread_id),
    )
    return ChatResponse(
        thread_id=thread_id,
        reply=extract_last_ai_text(result["messages"]),
    )
