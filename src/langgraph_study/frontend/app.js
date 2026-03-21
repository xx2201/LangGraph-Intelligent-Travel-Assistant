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
    threadIdEl.textContent = "not-created";
  }
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) {
    outputBox.textContent = "Please enter a question first.";
    return;
  }

  sendBtn.disabled = true;
  outputBox.textContent = "Calling LangGraph runtime...";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        thread_id: getThreadId(),
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    setThreadId(data.thread_id);
    outputBox.textContent = data.reply;
  } catch (error) {
    outputBox.textContent = `Request failed: ${error}`;
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
newThreadBtn.addEventListener("click", () => {
  setThreadId("");
  outputBox.textContent = "A new thread has been created locally.";
});

setThreadId(getThreadId());
