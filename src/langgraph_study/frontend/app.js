const threadIdEl = document.getElementById("threadId");
const sendBtn = document.getElementById("sendBtn");
const newThreadBtn = document.getElementById("newThreadBtn");
const messageInput = document.getElementById("messageInput");
const chatHistory = document.getElementById("chatHistory");
const sessionList = document.getElementById("sessionList");
const processTimeline = document.getElementById("processTimeline");

const ACTIVE_THREAD_KEY = "langgraph-active-thread-id";

let threads = [];
let activeThreadId = localStorage.getItem(ACTIVE_THREAD_KEY) || "";
let activeMessages = [];
let activeProcessEvents = [];
let activeAssistantContentEl = null;

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

function resetProcessTimeline(message = "发送一条消息后，这里会动态出现本轮 Agent 执行步骤。") {
  activeProcessEvents = [];
  renderProcessTimeline(message);
}

function formatProcessStatus(status) {
  if (status === "running") {
    return "running";
  }
  if (status === "done") {
    return "done";
  }
  return "info";
}

function upsertProcessEvent(event) {
  const nextEvent = {
    key: event.key,
    title: event.title,
    detail: event.detail,
    stage: event.stage,
    status: formatProcessStatus(event.status),
  };

  const index = activeProcessEvents.findIndex((item) => item.key === event.key);
  if (index >= 0) {
    activeProcessEvents = activeProcessEvents.map((item, itemIndex) =>
      itemIndex === index ? { ...item, ...nextEvent } : item,
    );
  } else {
    activeProcessEvents = [...activeProcessEvents, nextEvent];
  }

  renderProcessTimeline();
}

function renderSessionList() {
  sessionList.innerHTML = "";

  if (!threads.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "还没有会话。";
    sessionList.appendChild(empty);
    return;
  }

  threads.forEach((thread) => {
    const item = document.createElement("article");
    item.className = `session-item${thread.thread_id === activeThreadId ? " active" : ""}`;
    item.dataset.threadId = thread.thread_id;

    const row = document.createElement("div");
    row.className = "session-item-row";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item-select";
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

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        await deleteThread(thread.thread_id);
      } catch (error) {
        window.alert(`删除会话失败: ${error}`);
      }
    });

    row.append(button, deleteButton);
    item.appendChild(row);
    sessionList.appendChild(item);
  });
}

function createMessageElement(message) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${message.role}`;

  const role = document.createElement("span");
  role.className = "message-role";
  role.textContent = message.role === "user" ? "You" : "Agent";

  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.content;

  wrapper.append(role, content);
  return { wrapper, content };
}

function clearChatEmptyState() {
  const empty = chatHistory.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function scrollChatToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderChatHistory() {
  chatHistory.innerHTML = "";
  activeAssistantContentEl = null;
  updateThreadDisplay(activeThreadId);

  if (!activeThreadId) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "请选择左侧会话，或先创建一个新会话。";
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
    const { wrapper, content } = createMessageElement(message);
    if (message.role === "assistant") {
      activeAssistantContentEl = content;
    }
    chatHistory.appendChild(wrapper);
  });

  scrollChatToBottom();
}

function renderProcessTimeline(emptyText = "发送一条消息后，这里会动态出现本轮 Agent 执行步骤。") {
  processTimeline.innerHTML = "";

  if (!activeProcessEvents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = emptyText;
    processTimeline.appendChild(empty);
    return;
  }

  const rail = document.createElement("div");
  rail.className = "timeline-rail";

  activeProcessEvents.forEach((event) => {
    const article = document.createElement("article");
    article.className = `timeline-step ${event.status}`;

    const head = document.createElement("div");
    head.className = "timeline-step-head";

    const title = document.createElement("h3");
    title.className = "timeline-title";
    title.textContent = event.title;

    const badge = document.createElement("span");
    badge.className = "timeline-badge";
    badge.textContent = event.status;

    const detail = document.createElement("p");
    detail.className = "timeline-detail";
    detail.textContent = event.detail;

    head.append(title, badge);
    article.append(head, detail);
    rail.appendChild(article);
  });

  processTimeline.appendChild(rail);
  processTimeline.scrollTop = processTimeline.scrollHeight;
}

function renderApp() {
  renderSessionList();
  renderChatHistory();
  renderProcessTimeline();
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
    resetProcessTimeline();
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
  resetProcessTimeline("切换会话后，这里会显示下一轮对话的动态执行过程。");
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
  resetProcessTimeline("新会话已创建。发送一条消息后，这里会显示 Agent 动态过程。");
  renderApp();
}

async function deleteThread(threadId) {
  const thread = threads.find((item) => item.thread_id === threadId);
  const title = thread?.title || threadId;
  if (!window.confirm(`确认删除会话“${title}”吗？这会同时删除持久化上下文。`)) {
    return;
  }

  const response = await fetch(`/api/threads/${threadId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  await refreshThreads();
  const deletedCurrent = activeThreadId === threadId;
  if (!deletedCurrent) {
    renderApp();
    return;
  }

  if (!threads.length) {
    setActiveThreadId("");
    activeMessages = [];
    resetProcessTimeline("会话已删除。新建会话后，这里会显示新的执行过程。");
    renderApp();
    return;
  }

  await selectThread(threads[0].thread_id, false);
}

function appendUserMessage(content) {
  activeMessages = [...activeMessages, { role: "user", content }];
  clearChatEmptyState();
  const { wrapper } = createMessageElement({ role: "user", content });
  chatHistory.appendChild(wrapper);
  scrollChatToBottom();
}

function appendAssistantPlaceholder() {
  activeMessages = [...activeMessages, { role: "assistant", content: "" }];
  clearChatEmptyState();
  const { wrapper, content } = createMessageElement({ role: "assistant", content: "" });
  activeAssistantContentEl = content;
  chatHistory.appendChild(wrapper);
  scrollChatToBottom();
}

function appendAssistantDelta(content) {
  if (!activeMessages.length || activeMessages[activeMessages.length - 1].role !== "assistant") {
    appendAssistantPlaceholder();
  }
  const lastIndex = activeMessages.length - 1;
  const lastMessage = activeMessages[lastIndex];
  const nextMessage = { ...lastMessage, content: `${lastMessage.content}${content}` };
  activeMessages = [
    ...activeMessages.slice(0, lastIndex),
    nextMessage,
  ];
  if (activeAssistantContentEl) {
    activeAssistantContentEl.textContent = nextMessage.content;
    scrollChatToBottom();
    return;
  }
  renderChatHistory();
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

  const currentThreadId = activeThreadId;
  appendUserMessage(userMessage);
  resetProcessTimeline("Agent 已接收问题，正在启动本轮执行。");
  messageInput.value = "";
  sendBtn.disabled = true;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        thread_id: currentThreadId,
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
      if (event.type === "process") {
        upsertProcessEvent(event);
      }
      if (event.type === "assistant_start") {
        appendAssistantPlaceholder();
      }
      if (event.type === "assistant_delta") {
        appendAssistantDelta(event.content);
      }
      if (event.type === "done") {
        await refreshThreads();
        activeMessages = event.messages;
        renderSessionList();
        renderChatHistory();
      }
    });
  } catch (error) {
    upsertProcessEvent({
      key: "graph_error",
      stage: "graph",
      status: "done",
      title: "本轮失败",
      detail: `请求失败: ${error}`,
    });
    appendAssistantDelta(`请求失败: ${error}`);
  } finally {
    sendBtn.disabled = false;
    messageInput.focus();
  }
}

messageInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    await sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);
newThreadBtn.addEventListener("click", createThread);

loadThreads().catch((error) => {
  chatHistory.innerHTML = `<div class="empty-state">初始化失败: ${error}</div>`;
});
