// Health Assistant chat client. Talks to POST /chat; the conversation state lives here, not on the server.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    chat: $("chat"), messages: $("messages"), welcome: $("welcome"), form: $("composer"), input: $("input"),
    send: $("send"), counter: $("counter"), status: $("status"), statusText: $("status-text"),
    notice: $("notice"), noticeClose: $("notice-close"), newChat: $("new-chat"), theme: $("theme"),
  };
  const MAX = 500;
  const STORE = "health-assistant:conversation";
  const RETRY_FOR_MS = 120000; // Render's free tier can take a minute or two to wake up
  let state = null;
  let transcript = []; // [{role, text, kind, quickReplies}]
  let busy = false;

  // ---------- storage (best effort: private windows may block it) ----------
  const store = {
    get(key, area = sessionStorage) { try { return JSON.parse(area.getItem(key)); } catch { return null; } },
    set(key, value, area = sessionStorage) { try { area.setItem(key, JSON.stringify(value)); } catch { /* ignore */ } },
    del(key, area = sessionStorage) { try { area.removeItem(key); } catch { /* ignore */ } },
  };

  // ---------- rendering ----------
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function formatText(text) {
    // Paragraphs on blank lines, **bold** markers; everything else is escaped.
    return text.split(/\n{2,}/).map((p) =>
      `<p>${escapeHtml(p).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>")}</p>`).join("");
  }

  function avatar() {
    const a = document.createElement("div");
    a.className = "avatar";
    a.setAttribute("aria-hidden", "true");
    a.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M10 3h4v7h7v4h-7v7h-4v-7H3v-4h7z"/></svg>';
    return a;
  }

  function alertBody(kind, text) {
    const title = kind === "crisis" ? "You're not alone" : "This may be an emergency";
    const actions = kind === "crisis"
      ? '<a href="tel:988">Call 988</a><a class="secondary" href="sms:988">Text 988</a>'
      : '<a href="tel:911">Call 911</a>';
    return `<div class="alert-title"><span aria-hidden="true">⚠️</span>${title}</div>${formatText(text)}` +
           `<div class="alert-actions">${actions}</div>`;
  }

  function addMessage(msg, { animate = true } = {}) {
    els.welcome.hidden = true;
    const row = document.createElement("div");
    row.className = `row ${msg.role}`;
    if (!animate) row.style.animation = "none";
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (msg.role === "user") {
      bubble.textContent = msg.text;
      row.append(bubble);
    } else {
      const col = document.createElement("div");
      col.className = "bot-col";
      if (msg.kind === "emergency" || msg.kind === "crisis") {
        bubble.classList.add("alert");
        bubble.setAttribute("role", "alert");
        bubble.innerHTML = alertBody(msg.kind, msg.text);
      } else {
        bubble.innerHTML = formatText(msg.text);
      }
      col.append(bubble);
      if (msg.quickReplies && msg.quickReplies.length) {
        const chips = document.createElement("div");
        chips.className = "chips" + (msg.used ? " used" : "");
        for (const label of msg.quickReplies) {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "chip";
          b.textContent = label;
          b.addEventListener("click", () => send(label));
          chips.append(b);
        }
        col.append(chips);
      }
      row.append(avatar(), col);
    }
    els.messages.append(row);
    scrollDown();
    return row;
  }

  function addTyping() {
    const row = document.createElement("div");
    row.className = "row bot";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<span class="typing" aria-label="Assistant is typing"><span></span><span></span><span></span></span>';
    row.append(avatar(), bubble);
    els.messages.append(row);
    scrollDown();
    return row;
  }

  function addNote(text) {
    const p = document.createElement("p");
    p.className = "system-note";
    p.textContent = text;
    els.messages.append(p);
    scrollDown();
    return p;
  }

  function scrollDown() { els.chat.scrollTop = els.chat.scrollHeight; }

  function retireChips() {
    els.messages.querySelectorAll(".chips:not(.used)").forEach((c) => c.classList.add("used"));
    transcript.forEach((m) => { if (m.quickReplies) m.used = true; });
  }

  function save() { store.set(STORE, { state, transcript }); }

  // ---------- server status ----------
  function setStatus(kind, text) {
    els.status.className = `status ${kind}`;
    els.statusText.textContent = text;
  }

  async function checkHealth() {
    const started = Date.now();
    setStatus("waking", "Connecting…");
    while (Date.now() - started < RETRY_FOR_MS) {
      try {
        const r = await fetch("/health", { cache: "no-store" });
        if (r.ok && (await r.json()).model_loaded) { setStatus("online", "Online · general guidance only"); return; }
      } catch { /* server asleep or starting */ }
      setStatus("waking", "Waking up the server…");
      await new Promise((res) => setTimeout(res, 3000));
    }
    setStatus("offline", "Offline · try again shortly");
  }

  // ---------- sending ----------
  async function postChat(message) {
    const started = Date.now();
    let note = null;
    for (;;) {
      try {
        const r = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, state }),
        });
        if (r.ok) { if (note) note.remove(); return await r.json(); }
        if (r.status === 422) { if (note) note.remove(); throw new Error("invalid"); }
      } catch (e) {
        if (e.message === "invalid") throw e;
      }
      if (Date.now() - started > RETRY_FOR_MS) { if (note) note.remove(); throw new Error("timeout"); }
      if (!note) note = addNote("The server is waking up, this can take a minute on the free tier…");
      setStatus("waking", "Waking up the server…");
      await new Promise((res) => setTimeout(res, 3000));
    }
  }

  async function send(text) {
    text = (text || "").trim().slice(0, MAX);
    if (!text || busy) return;
    busy = true;
    retireChips();
    const userMsg = { role: "user", text };
    transcript.push(userMsg);
    addMessage(userMsg);
    els.input.value = "";
    autosize();
    updateComposer();

    const typing = addTyping();
    const minDelay = new Promise((res) => setTimeout(res, 450)); // a short pause reads more naturally
    try {
      const [data] = await Promise.all([postChat(text), minDelay]);
      typing.remove();
      state = data.state;
      const botMsg = { role: "bot", text: data.reply, kind: data.kind, quickReplies: data.quick_replies };
      transcript.push(botMsg);
      addMessage(botMsg);
      setStatus("online", "Online · general guidance only");
    } catch (e) {
      typing.remove();
      addNote(e.message === "invalid" ? "That message couldn't be sent. Please try a shorter message."
                                      : "Couldn't reach the server. Please try again in a moment.");
      setStatus("offline", "Offline · try again shortly");
    } finally {
      busy = false;
      save();
      updateComposer();
      els.input.focus();
    }
  }

  // ---------- composer ----------
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 140) + "px";
  }

  function updateComposer() {
    const n = els.input.value.trim().length;
    els.send.disabled = busy || n === 0;
    els.counter.textContent = els.input.value.length > MAX - 100 ? `${els.input.value.length}/${MAX}` : "";
  }

  els.input.addEventListener("input", () => { autosize(); updateComposer(); });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(els.input.value); }
  });
  els.form.addEventListener("submit", (e) => { e.preventDefault(); send(els.input.value); });
  document.querySelectorAll(".suggestion").forEach((b) => b.addEventListener("click", () => send(b.dataset.text)));

  // ---------- header actions ----------
  els.newChat.addEventListener("click", () => {
    state = null;
    transcript = [];
    store.del(STORE);
    els.messages.replaceChildren();
    els.welcome.hidden = false;
    els.input.focus();
  });

  function applyTheme(theme) {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }
  els.theme.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    applyTheme(next);
    store.set("health-assistant:theme", next, localStorage);
  });

  els.noticeClose.addEventListener("click", () => {
    els.notice.hidden = true;
    store.set("health-assistant:notice-dismissed", true, sessionStorage);
  });

  // ---------- start ----------
  applyTheme(store.get("health-assistant:theme", localStorage));
  if (store.get("health-assistant:notice-dismissed")) els.notice.hidden = true;
  const saved = store.get(STORE);
  if (saved && Array.isArray(saved.transcript) && saved.transcript.length) {
    state = saved.state;
    transcript = saved.transcript;
    transcript.forEach((m) => addMessage(m, { animate: false }));
  }
  updateComposer();
  checkHealth();
})();
