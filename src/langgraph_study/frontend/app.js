const threadIdEl = document.getElementById("threadId");
const sendBtn = document.getElementById("sendBtn");
const newThreadBtn = document.getElementById("newThreadBtn");
const messageInput = document.getElementById("messageInput");
const chatHistory = document.getElementById("chatHistory");
const sessionList = document.getElementById("sessionList");
const stateMessageCount = document.getElementById("stateMessageCount");
const stateUserCount = document.getElementById("stateUserCount");
const stateAssistantCount = document.getElementById("stateAssistantCount");
const stateToolCallCount = document.getElementById("stateToolCallCount");
const stateRouteHint = document.getElementById("stateRouteHint");
const stateQueryContext = document.getElementById("stateQueryContext");

const ACTIVE_THREAD_KEY = "langgraph-active-thread-id";

let threads = [];
let activeThreadId = localStorage.getItem(ACTIVE_THREAD_KEY) || "";
let activeMessages = [];
let activeState = {
  query_context: {},
  message_count: 0,
  user_message_count: 0,
  assistant_message_count: 0,
  tool_call_count: 0,
  next_route_hint: "assistant",
};

function setActiveThreadId(threadId) {
  activeThreadId = threadId;
  if (threadId) {
    localStorage.setItem(ACTIVE_THREAD_KEY, threadId);
  } else {
    localStorage.removeItem(ACTIVE_THREAD_KEY);
  }
}

function updateThreadDisplay(threadId) {
  threadIdEl.textContent = threadId || "not-created";
}

function renderStatePanel() {
  stateMessageCount.textContent = String(activeState.message_count || 0);
  stateUserCount.textContent = String(activeState.user_message_count || 0);
  stateAssistantCount.textContent = String(activeState.assistant_message_count || 0);
  stateToolCallCount.textContent = String(activeState.tool_call_count || 0);
  stateRouteHint.textContent = activeState.next_route_hint || "assistant";
  stateQueryContext.textContent = JSON.stringify(activeState.query_context || {}, null, 2);
}

function renderSessionList() {
  sessionList.innerHTML = "";

  if (!threads.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "后端还没有会话。点击“新建会话”开始。";
    sessionList.appendChild(empty);
    return;
  }

  threads.forEach((thread) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item${thread.thread_id === activeThreadId ? " active" : ""}`;
    button.dataset.threadId = thread.thread_id;

    const title = document.createElement("div");
    title.className = "session-title";
    title.textContent = thread.title;

    const meta = document.createElement("div");
    meta.className = "session-meta";
    meta.textContent = `thread_id: ${thread.thread_id}`;

    const preview = document.createElement("div");
    preview.className = "session-preview";
    preview.textContent = thread.last_user_message || thread.last_assistant_message || "暂无消息";

    button.append(title, meta, preview);
    button.addEventListener("click", async () => {
      await selectThread(thread.thread_id);
    });
    sessionList.appendChild(button);
  });
}

function renderChatHistory() {
  chatHistory.innerHTML = "";
  updateThreadDisplay(activeThreadId);

  if (!activeThreadId) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "请选择左侧会话，或先新建一个会话。";
    chatHistory.appendChild(empty);
    return;
  }

  if (!activeMessages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "当前会话还没有消息。先在下方输入问题。";
    chatHistory.appendChild(empty);
    return;
  }

  activeMessages.forEach((message) => {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${message.role}`;

    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "User" : "Assistant";

    const content = document.createElement("div");
    content.textContent = message.content;

    wrapper.append(role, content);
    chatHistory.appendChild(wrapper);
  });

  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderApp() {
  renderSessionList();
  renderChatHistory();
  renderStatePanel();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function refreshThreads() {
  threads = await fetchJson("/api/threads");
}

async function loadThreads() {
  await refreshThreads();

  if (!threads.length) {
    setActiveThreadId("");
    activeMessages = [];
    renderApp();
    return;
  }

  const found = threads.find((thread) => thread.thread_id === activeThreadId);
  const nextThreadId = found ? activeThreadId : threads[0].thread_id;
  await selectThread(nextThreadId, false);
}

async function selectThread(threadId, rerenderList = true) {
  const detail = await fetchJson(`/api/threads/${threadId}`);
  setActiveThreadId(detail.thread.thread_id);
  activeMessages = detail.messages;
  activeState = detail.state;
  if (rerenderList) {
    await refreshThreads();
  }
  renderApp();
}

async function createThread() {
  const thread = await fetchJson("/api/threads", { method: "POST" });
  await refreshThreads();
  setActiveThreadId(thread.thread_id);
  activeMessages = [];
  activeState = {
    query_context: {},
    message_count: 0,
    user_message_count: 0,
    assistant_message_count: 0,
    tool_call_count: 0,
    next_route_hint: "assistant",
  };
  renderApp();
}

function appendOptimisticUserMessage(content) {
  activeMessages = [...activeMessages, { role: "user", content }];
  activeState = {
    ...activeState,
    message_count: (activeState.message_count || 0) + 1,
    user_message_count: (activeState.user_message_count || 0) + 1,
  };
  renderApp();
}

function startAssistantMessage() {
  activeMessages = [...activeMessages, { role: "assistant", content: "" }];
  renderApp();
}

function appendAssistantDelta(content) {
  if (!activeMessages.length) {
    startAssistantMessage();
  }
  const lastMessage = activeMessages[activeMessages.length - 1];
  if (!lastMessage || lastMessage.role !== "assistant") {
    startAssistantMessage();
  }
  activeMessages = activeMessages.map((message, index) => {
    if (index !== activeMessages.length - 1) {
      return message;
    }
    return {
      ...message,
      content: `${message.content}${content}`,
    };
  });
  renderApp();
}

async function consumeNdjsonStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      await onEvent(JSON.parse(line));
    }
  }

  if (buffer.trim()) {
    await onEvent(JSON.parse(buffer));
  }
}

async function sendMessage() {
  const userMessage = messageInput.value.trim();
  if (!userMessage) {
    return;
  }

  if (!activeThreadId) {
    await createThread();
  }

  const sentThreadId = activeThreadId;
  appendOptimisticUserMessage(userMessage);
  messageInput.value = "";
  sendBtn.disabled = true;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        thread_id: sentThreadId,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    await consumeNdjsonStream(response, async (event) => {
      if (event.type === "thread") {
        setActiveThreadId(event.thread_id);
        updateThreadDisplay(event.thread_id);
      }
      if (event.type === "assistant_start") {
        startAssistantMessage();
      }
      if (event.type === "assistant_delta") {
        appendAssistantDelta(event.content);
      }
      if (event.type === "done") {
        await refreshThreads();
        activeMessages = event.messages;
        activeState = event.state;
        renderApp();
      }
    });
  } catch (error) {
    appendAssistantDelta(`请求失败：${error}`);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
newThreadBtn.addEventListener("click", createThread);

loadThreads().catch((error) => {
  chatHistory.innerHTML = `<div class="empty-state">初始化失败：${error}</div>`;
});
