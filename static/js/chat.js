/* ImageBot – Chat UI */

const chatThread   = document.getElementById('chatThread');
const chatWelcome  = document.getElementById('chatWelcome');
const chatInput    = document.getElementById('chatInput');
const sendBtn      = document.getElementById('sendBtn');
const sendLabel    = document.getElementById('sendLabel');
const sendSpinner  = document.getElementById('sendSpinner');
const attachBtn    = document.getElementById('attachBtn');
const attachInput  = document.getElementById('attachInput');
const attachPreview = document.getElementById('attachPreview');
const attachPreviewImg = document.getElementById('attachPreviewImg');
const attachRemove = document.getElementById('attachRemove');
const statusDot    = document.getElementById('statusDot');
const statusText   = document.getElementById('statusText');
const footerModel  = document.getElementById('footerModel');

/** @type {{ role: string, content: string, imageDataUrl?: string | null }[]} */
let messages = [];
let pendingFile = null;

function scrollToBottom() {
  chatThread.scrollTop = chatThread.scrollHeight;
}

function hideWelcome() {
  if (chatWelcome) chatWelcome.style.display = 'none';
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  attachBtn.disabled = busy;
  sendLabel.style.display = busy ? 'none' : '';
  sendSpinner.style.display = busy ? 'inline-block' : 'none';
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    footerModel.textContent = data.ollama_model || '';
    if (data.ollama_available) {
      statusDot.className = 'status-dot ok';
      statusText.textContent = `${data.vision_backend || ''} · ${data.ollama_model || ''}`;
    } else {
      statusDot.className = 'status-dot err';
      statusText.textContent = 'Ollama offline';
    }
  } catch {
    statusDot.className = 'status-dot err';
    statusText.textContent = 'Server offline';
  }
}
checkStatus();
setInterval(checkStatus, 30_000);

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderMessage(role, content, imageDataUrl, options = {}) {
  const { streaming } = options;
  const wrap = document.createElement('div');
  wrap.className = `chat-msg chat-msg-${role}${streaming ? ' streaming' : ''}`;

  const label = document.createElement('div');
  label.className = 'chat-msg-label';
  label.textContent = role === 'user' ? 'You' : 'Assistant';

  if (imageDataUrl && role === 'user') {
    const img = document.createElement('img');
    img.className = 'chat-msg-img';
    img.src = imageDataUrl;
    img.alt = 'Attached';
    wrap.appendChild(label);
    wrap.appendChild(img);
  } else {
    wrap.appendChild(label);
  }

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.innerHTML = escapeHtml(content);
  wrap.appendChild(bubble);

  chatThread.appendChild(wrap);
  scrollToBottom();
  return { wrap, bubble };
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function clearPendingAttachment() {
  pendingFile = null;
  attachInput.value = '';
  attachPreview.style.display = 'none';
  attachPreviewImg.removeAttribute('src');
}

attachBtn.addEventListener('click', () => attachInput.click());
attachInput.addEventListener('change', () => {
  const f = attachInput.files && attachInput.files[0];
  if (!f || !f.type.startsWith('image/')) {
    clearPendingAttachment();
    return;
  }
  pendingFile = f;
  attachPreviewImg.src = URL.createObjectURL(f);
  attachPreview.style.display = 'inline-block';
});
attachRemove.addEventListener('click', clearPendingAttachment);

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

function historyForApi() {
  return messages.map(m => ({ role: m.role, content: m.content }));
}

async function sendMessage() {
  const text = chatInput.value.trim();
  const hasImg = !!pendingFile;
  if (!text && !hasImg) return;

  hideWelcome();
  const fileSnapshot = pendingFile;
  /** @type {string | null} */
  const thumbDataUrl = hasImg ? await fileToDataUrl(fileSnapshot) : null;
  const displayText = text || (hasImg ? '(image only)' : '');

  renderMessage('user', displayText, thumbDataUrl);
  chatInput.value = '';
  chatInput.style.height = 'auto';

  const form = new FormData();
  form.append('history', JSON.stringify(historyForApi()));
  form.append('message', text);
  if (fileSnapshot) form.append('image', fileSnapshot, fileSnapshot.name);

  clearPendingAttachment();

  const { wrap, bubble } = renderMessage('assistant', '', null, { streaming: true });
  setBusy(true);

  try {
    const resp = await fetch('/api/chat/stream', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      wrap.remove();
      renderMessage('assistant', `Error: ${err.error || resp.statusText}`, null);
      setBusy(false);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullReply = '';
    let userContentForLog = text || '(image only)';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      let sseEvent = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          sseEvent = line.slice(7).trim();
          continue;
        }
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        let parsed;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          wrap.classList.remove('streaming');
          bubble.textContent = `Error: ${parsed.error}`;
          setBusy(false);
          sseEvent = '';
          return;
        }

        if (sseEvent === 'meta' && parsed.user_content) {
          userContentForLog = parsed.user_content;
        }

        if (parsed.token !== undefined) {
          fullReply += parsed.token;
          bubble.textContent = fullReply;
          scrollToBottom();
        }

        sseEvent = '';
      }
    }

    wrap.classList.remove('streaming');

    messages.push({
      role: 'user',
      content: userContentForLog,
      imageDataUrl: thumbDataUrl || null,
    });
    messages.push({ role: 'assistant', content: fullReply, imageDataUrl: null });
  } catch (e) {
    wrap.classList.remove('streaming');
    bubble.textContent = `Network error: ${e.message}`;
  } finally {
    setBusy(false);
    scrollToBottom();
  }
}
