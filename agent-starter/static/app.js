const messages = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const send = document.getElementById("send");

function addMessage(role, text, extraClass = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role} ${extraClass}`.trim();
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text; // textContent: never render model/document text as HTML
  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
  return wrapper;
}

function addSources(wrapper, sources) {
  if (!sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  const title = document.createElement("div");
  title.textContent = "Sources:";
  box.appendChild(title);
  for (const source of sources) {
    const line = document.createElement("div");
    line.textContent = `📄 ${source.document}` + (source.page ? ` - Page ${source.page}` : "");
    box.appendChild(line);
  }
  wrapper.appendChild(box);
}

async function ask(question) {
  addMessage("user", question);
  const pending = addMessage("assistant", "Thinking", "loading");
  send.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const data = await response.json().catch(() => ({}));
    pending.remove();
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Something went wrong.";
      addMessage("assistant", detail, "error");
      return;
    }
    addSources(addMessage("assistant", data.answer), data.sources);
  } catch {
    pending.remove();
    addMessage("assistant", "Could not reach the server. Please try again.", "error");
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});
