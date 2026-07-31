/* StructRAG frontend.
 *
 * Holds no secrets. The invite code is exchanged for a short-lived session
 * token and never stored. Limits shown here mirror what the server enforces;
 * the server is the boundary, since a visitor can edit this file at will.
 *
 * Server-provided strings reach the DOM only through textContent, so a filename
 * or answer containing markup is displayed rather than parsed.
 */

"use strict";

(function () {
  const CFG = window.STRUCTRAG_CONFIG || {};
  const API = (CFG.API_BASE || "").replace(/\/+$/, "");
  const POLL_MS = CFG.POLL_INTERVAL_MS || 1500;
  const UPLOAD_TIMEOUT_MS = CFG.UPLOAD_TIMEOUT_MS || 240000;

  // sessionStorage, not localStorage: the token dies with the tab.
  const TOKEN_KEY = "structrag_token";

  const el = (id) => document.getElementById(id);

  const ui = {
    gate: el("gate"),
    gateForm: el("gate-form"),
    gateSubmit: el("gate-submit"),
    gateError: el("gate-error"),
    invite: el("invite"),
    app: el("app"),
    signout: el("signout"),
    quota: el("quota"),
    dropzone: el("dropzone"),
    dropzoneLimits: el("dropzone-limits"),
    fileInput: el("file-input"),
    doclist: el("doclist"),
    uploadError: el("upload-error"),
    transcript: el("transcript"),
    emptyState: el("empty-state"),
    askForm: el("ask-form"),
    question: el("question"),
    askSubmit: el("ask-submit"),
    composerHint: el("composer-hint"),
  };

  let limits = { max_query_chars: 500, max_pdf_pages: 50, max_upload_mb: 10 };
  let readyDocs = 0;
  let streaming = false;

  // ── helpers ───────────────────────────────────────────────────────────────

  const token = () => sessionStorage.getItem(TOKEN_KEY);

  function showError(node, message) {
    node.textContent = message;
    node.hidden = false;
  }

  function clearError(node) {
    node.textContent = "";
    node.hidden = true;
  }

  /** Fetch with the session token attached, handling expiry centrally. */
  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    const t = token();
    if (t) headers["Authorization"] = "Bearer " + t;

    const response = await fetch(API + path, Object.assign({}, options, { headers }));

    if (response.status === 401) {
      signOut("Your session expired. Enter the invite code again.");
      throw new Error("session expired");
    }
    return response;
  }

  /** Pull a human-readable message out of an error response. */
  async function errorMessage(response, fallback) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return body.detail;
      // FastAPI validation errors arrive as an array of objects.
      if (Array.isArray(body.detail) && body.detail.length) {
        return body.detail[0].msg || fallback;
      }
    } catch (_) { /* not JSON */ }
    return fallback;
  }

  // ── session ───────────────────────────────────────────────────────────────

  ui.gateForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError(ui.gateError);

    const code = ui.invite.value.trim();
    if (!code) return;

    ui.gateSubmit.disabled = true;
    ui.gateSubmit.textContent = "Checking…";

    try {
      const response = await fetch(API + "/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ invite_code: code }),
      });

      if (!response.ok) {
        showError(ui.gateError, await errorMessage(response, "That code was not accepted."));
        return;
      }

      const body = await response.json();
      sessionStorage.setItem(TOKEN_KEY, body.token);
      if (body.limits) limits = Object.assign(limits, body.limits);

      // Clear the field so the code is not left sitting in the DOM.
      ui.invite.value = "";
      enterApp();
    } catch (_) {
      showError(ui.gateError, "Could not reach the server. It may be waking up — try again in a moment.");
    } finally {
      ui.gateSubmit.disabled = false;
      ui.gateSubmit.textContent = "Continue";
    }
  });

  function enterApp() {
    ui.gate.hidden = true;
    ui.app.hidden = false;
    ui.signout.hidden = false;
    ui.dropzoneLimits.textContent =
      `PDF · up to ${limits.max_upload_mb} MB · ${limits.max_pdf_pages} pages`;
    updateHint();
    refreshSession();
    ui.question.focus();
  }

  function signOut(message) {
    sessionStorage.removeItem(TOKEN_KEY);
    readyDocs = 0;
    ui.app.hidden = true;
    ui.gate.hidden = false;
    ui.signout.hidden = true;
    ui.quota.hidden = true;
    ui.doclist.replaceChildren();
    if (message) showError(ui.gateError, message);
  }

  ui.signout.addEventListener("click", () => signOut(null));

  async function refreshSession() {
    try {
      const response = await api("/api/session");
      if (!response.ok) return;
      const state = await response.json();
      readyDocs = state.documents || 0;
      ui.quota.textContent =
        `${state.queries_remaining} questions · ${state.uploads_remaining} uploads left`;
      ui.quota.hidden = false;
      updateAskEnabled();
    } catch (_) { /* transient; the next action will surface it */ }
  }

  // ── upload ────────────────────────────────────────────────────────────────

  ui.dropzone.addEventListener("click", () => ui.fileInput.click());
  ui.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      ui.fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((name) =>
    ui.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      ui.dropzone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach((name) =>
    ui.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      ui.dropzone.classList.remove("dragover");
    })
  );

  ui.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) upload(file);
  });

  ui.fileInput.addEventListener("change", () => {
    const file = ui.fileInput.files && ui.fileInput.files[0];
    if (file) upload(file);
    ui.fileInput.value = "";   // allow re-selecting the same file
  });

  async function upload(file) {
    clearError(ui.uploadError);

    // Saves a round trip only. The server re-checks size, magic bytes, page
    // count and active content.
    if (file.size > limits.max_upload_mb * 1024 * 1024) {
      showError(ui.uploadError,
        `That file is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${limits.max_upload_mb} MB.`);
      return;
    }

    const row = addDocRow(file.name);
    ui.dropzone.classList.add("busy");

    try {
      const form = new FormData();
      form.append("file", file, file.name);

      const response = await api("/api/upload", { method: "POST", body: form });

      if (!response.ok) {
        const message = await errorMessage(response, "That file could not be uploaded.");
        setDocRow(row, "failed", message);
        showError(ui.uploadError, message);
        return;
      }

      const job = await response.json();
      setDocRow(row, "working", `${job.page_count} pages · parsing`);
      await pollJob(job.job_id, row);
    } catch (error) {
      if (error.message !== "session expired") {
        setDocRow(row, "failed", "Upload failed.");
        showError(ui.uploadError, "Upload failed. Check your connection and try again.");
      }
    } finally {
      ui.dropzone.classList.remove("busy");
      refreshSession();
    }
  }

  async function pollJob(jobId, row) {
    const deadline = Date.now() + UPLOAD_TIMEOUT_MS;

    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));

      const response = await api("/api/jobs/" + encodeURIComponent(jobId));
      if (!response.ok) {
        setDocRow(row, "failed", "Lost track of this upload.");
        return;
      }

      const job = await response.json();

      if (job.status === "ready") {
        setDocRow(row, "ready", `${job.pages} pages · ${job.chunks} chunks`);
        readyDocs += 1;
        updateAskEnabled();
        ui.question.focus();
        return;
      }

      if (job.status === "failed") {
        setDocRow(row, "failed", job.error || "Could not process this document.");
        showError(ui.uploadError,
          (job.error || "Could not process this document.") +
          " If it is a scanned document, OCR is not available on this demo.");
        return;
      }

      setDocRow(row, "working", `${job.pages} pages · ${job.status}`);
    }

    setDocRow(row, "failed", "Timed out.");
  }

  function addDocRow(name) {
    ui.doclist.appendChild(document.createElement("li"));
    const row = ui.doclist.lastElementChild;

    const status = document.createElement("span");
    status.className = "doc-status working";
    status.textContent = "◌";

    const label = document.createElement("span");
    label.className = "doc-name";
    // textContent, so a filename containing markup is shown, not executed.
    label.textContent = name;

    const meta = document.createElement("span");
    meta.className = "doc-meta";
    meta.textContent = "uploading…";
    label.appendChild(meta);

    row.appendChild(status);
    row.appendChild(label);
    return row;
  }

  function setDocRow(row, state, metaText) {
    const status = row.querySelector(".doc-status");
    const meta = row.querySelector(".doc-meta");
    status.className = "doc-status " + state;
    status.textContent = state === "ready" ? "✓" : state === "failed" ? "✗" : "◌";
    meta.textContent = metaText;
  }

  // ── composer ──────────────────────────────────────────────────────────────

  ui.question.addEventListener("input", () => {
    ui.question.style.height = "auto";
    ui.question.style.height = Math.min(ui.question.scrollHeight, 160) + "px";
    updateHint();
    updateAskEnabled();
  });

  ui.question.addEventListener("keydown", (event) => {
    // Enter sends, Shift+Enter makes a newline.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ui.askForm.requestSubmit();
    }
  });

  function updateHint() {
    const used = ui.question.value.length;
    const max = limits.max_query_chars;
    if (used > max * 0.8) {
      ui.composerHint.textContent = `${used} / ${max} characters`;
      ui.composerHint.classList.toggle("over", used > max);
    } else if (readyDocs === 0) {
      ui.composerHint.textContent = "Upload a document to start asking questions.";
      ui.composerHint.classList.remove("over");
    } else {
      ui.composerHint.textContent = "";
      ui.composerHint.classList.remove("over");
    }
  }

  function updateAskEnabled() {
    const text = ui.question.value.trim();
    ui.askSubmit.disabled =
      streaming || readyDocs === 0 || text.length === 0 ||
      text.length > limits.max_query_chars;
    updateHint();
  }

  // ── ask ───────────────────────────────────────────────────────────────────

  ui.askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = ui.question.value.trim();
    if (!question || streaming) return;

    streaming = true;
    updateAskEnabled();
    if (ui.emptyState) ui.emptyState.remove();

    const turn = addTurn(question);

    ui.question.value = "";
    ui.question.style.height = "auto";

    try {
      const response = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });

      if (!response.ok) {
        const message = await errorMessage(response, "That question could not be answered.");
        finishTurn(turn, message, "failed");
        return;
      }

      await readStream(response, turn);
    } catch (error) {
      if (error.message !== "session expired") {
        finishTurn(turn, "Lost connection to the server.", "failed");
      }
    } finally {
      streaming = false;
      updateAskEnabled();
      refreshSession();
      ui.question.focus();
    }
  });

  /* Reads the SSE stream.
   *
   * EventSource only issues GET and cannot send an Authorization header, while
   * /api/query is an authenticated POST — so frames are parsed off a fetch body
   * reader. They are blank-line separated and can split across chunks, hence
   * the buffer. */
  async function readStream(response, turn) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);

        for (const line of frame.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try {
            event = JSON.parse(line.slice(6));
          } catch (_) {
            continue;   // partial or malformed frame; skip it
          }
          handleEvent(event, turn);
        }
      }
    }

    turn.answer.classList.remove("cursor");
  }

  function handleEvent(event, turn) {
    switch (event.type) {
      case "token":
        turn.answer.textContent += event.content;
        scrollToBottom();
        break;

      case "refusal":
        turn.answer.textContent = event.content;
        turn.answer.classList.add("refused");
        break;

      case "citations":
        turn.citations = event.sources || {};
        break;

      case "error":
        turn.answer.textContent = event.content;
        turn.answer.classList.add("failed");
        break;

      case "done":
        turn.answer.classList.remove("cursor");
        renderMeta(turn, event);
        break;
    }
  }

  function addTurn(question) {
    const wrap = document.createElement("div");
    wrap.className = "turn";

    const q = document.createElement("p");
    q.className = "turn-q";
    q.textContent = question;

    const a = document.createElement("p");
    a.className = "turn-a cursor";

    wrap.appendChild(q);
    wrap.appendChild(a);
    ui.transcript.appendChild(wrap);
    scrollToBottom();

    return { wrap, answer: a, citations: null };
  }

  function finishTurn(turn, message, state) {
    turn.answer.classList.remove("cursor");
    turn.answer.textContent = message;
    if (state) turn.answer.classList.add(state);
  }

  function renderMeta(turn, event) {
    const meta = document.createElement("div");
    meta.className = "turn-meta";

    const cites = turn.citations || event.citations || {};
    for (const [filename, pages] of Object.entries(cites)) {
      const chip = document.createElement("span");
      chip.className = "cite";
      chip.textContent = `${filename} · p.${pages.join(", ")}`;
      meta.appendChild(chip);
    }

    if (!event.refused && typeof event.confidence === "number") {
      const score = document.createElement("span");
      const pct = Math.round(event.confidence * 100);
      score.className =
        "confidence " + (pct >= 70 ? "high" : pct >= 40 ? "mid" : "low");
      score.textContent = `confidence ${pct}%`;
      meta.appendChild(score);
    }

    if (typeof event.processing_time === "number") {
      const timing = document.createElement("span");
      timing.textContent = `${event.processing_time.toFixed(1)}s`;
      meta.appendChild(timing);
    }

    if (meta.childElementCount) {
      turn.wrap.appendChild(meta);
      scrollToBottom();
    }
  }

  function scrollToBottom() {
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
  }

  // ── boot ──────────────────────────────────────────────────────────────────

  // The backend origin is written in two places that cannot be derived from one
  // another: API_BASE in config.js, and connect-src in the CSP meta tag. If they
  // drift, the browser blocks every request before it leaves the page and the
  // only trace is a console violation — the UI just sits there looking broken.
  // Checking it here turns that into a message that names both files.
  function cspAllows(origin) {
    const meta = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
    if (!meta) return true;  // No meta CSP; a header may still apply. Do not guess.
    const directive = meta.content.split(";")
      .map((part) => part.trim())
      .find((part) => part.startsWith("connect-src"));
    if (!directive) return true;
    return directive.split(/\s+/).slice(1).some((src) => src === origin || src === "*");
  }

  if (!API) {
    showError(ui.gateError,
      "This page is not configured yet — set API_BASE in docs/config.js to your backend URL.");
    ui.gateSubmit.disabled = true;
  } else if (!cspAllows(new URL(API).origin)) {
    showError(ui.gateError,
      `Configuration mismatch: API_BASE is ${new URL(API).origin}, but the ` +
      "Content-Security-Policy in index.html does not allow it. Update the " +
      "connect-src entry to match, or the browser will block every request.");
    ui.gateSubmit.disabled = true;
  } else if (token()) {
    // Resume an existing tab session; the server rejects the token if stale.
    enterApp();
  }
})();
