const threadIdEl = document.getElementById("threadId");
const sendBtn = document.getElementById("sendBtn");
const newThreadBtn = document.getElementById("newThreadBtn");
const messageInput = document.getElementById("messageInput");
const chatHistory = document.getElementById("chatHistory");
const sessionList = document.getElementById("sessionList");

const ACTIVE_THREAD_KEY = "langgraph-active-thread-id";

let threads = [];
let activeThreadId = localStorage.getItem(ACTIVE_THREAD_KEY) || "";
let activeMessages = [];

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
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function loadThreads() {
  threads = await fetchJson("/api/threads");

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
  if (rerenderList) {
    threads = await fetchJson("/api/threads");
  }
  renderApp();
}

async function createThread() {
  const thread = await fetchJson("/api/threads", { method: "POST" });
  threads = await fetchJson("/api/threads");
  setActiveThreadId(thread.thread_id);
  activeMessages = [];
  renderApp();
}

async function sendMessage() {
  const userMessage = messageInput.value.trim();
  if (!userMessage) {
    return;
  }

  if (!activeThreadId) {
    await createThread();
  }

  sendBtn.disabled = true;

  try {
    await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        thread_id: activeThreadId,
      }),
    });

    messageInput.value = "";
    threads = await fetchJson("/api/threads");
    await selectThread(activeThreadId, false);
  } catch (error) {
    activeMessages = [
      ...activeMessages,
      { role: "user", content: userMessage },
      { role: "assistant", content: `请求失败：${error}` },
    ];
    renderApp();
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
newThreadBtn.addEventListener("click", createThread);

loadThreads().catch((error) => {
  chatHistory.innerHTML = `<div class="empty-state">初始化失败：${error}</div>`;
});
