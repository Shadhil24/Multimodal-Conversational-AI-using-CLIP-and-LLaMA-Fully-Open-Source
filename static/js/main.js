/* ============================================================
   ImageBot – Frontend Logic
   ============================================================ */

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewWrap = document.getElementById('previewWrap');
const previewImg  = document.getElementById('previewImg');
const btnRemove   = document.getElementById('btnRemove');
const fileInfo    = document.getElementById('fileInfo');
const fileName    = document.getElementById('fileName');
const fileSize    = document.getElementById('fileSize');
const btnAnalyze  = document.getElementById('btnAnalyze');
const btnLabel    = document.getElementById('btnLabel');
const btnSpinner  = document.getElementById('btnSpinner');

const placeholder   = document.getElementById('placeholder');
const clipSection   = document.getElementById('clipSection');
const labelGrid     = document.getElementById('labelGrid');
const descSection   = document.getElementById('descSection');
const descriptionBox = document.getElementById('descriptionBox');
const metaRow       = document.getElementById('metaRow');
const elapsedMs     = document.getElementById('elapsedMs');
const btnCopy       = document.getElementById('btnCopy');
const errorBox      = document.getElementById('errorBox');
const errorMsg      = document.getElementById('errorMsg');

const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

let selectedFile = null;

// ----------------------------------------------------------------
// Status check
// ----------------------------------------------------------------
async function checkStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    if (data.ollama_available) {
      statusDot.className  = 'status-dot ok';
      statusText.textContent = `Ollama ready · ${data.ollama_model}`;
    } else {
      statusDot.className  = 'status-dot err';
      statusText.textContent = 'Ollama offline';
    }
  } catch {
    statusDot.className  = 'status-dot err';
    statusText.textContent = 'Server offline';
  }
}
checkStatus();
setInterval(checkStatus, 30_000);

// ----------------------------------------------------------------
// File selection helpers
// ----------------------------------------------------------------
function formatBytes(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1024**2)    return `${(bytes/1024).toFixed(1)} KB`;
  return `${(bytes/1024**2).toFixed(2)} MB`;
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showError('Please select a valid image file.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropZone.style.display  = 'none';
    previewWrap.style.display = 'block';
    fileInfo.style.display    = 'flex';
    fileName.textContent      = file.name;
    fileSize.textContent      = formatBytes(file.size);
    btnAnalyze.disabled       = false;
    clearResults();
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dropZone.style.display    = '';
  previewWrap.style.display = 'none';
  fileInfo.style.display    = 'none';
  btnAnalyze.disabled       = true;
  clearResults();
}

function clearResults() {
  placeholder.style.display   = '';
  clipSection.style.display   = 'none';
  descSection.style.display   = 'none';
  errorBox.style.display      = 'none';
  metaRow.style.display       = 'none';
  labelGrid.innerHTML         = '';
  descriptionBox.innerHTML    = '';
}

// ----------------------------------------------------------------
// Drag & Drop
// ----------------------------------------------------------------
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
btnRemove.addEventListener('click', clearFile);

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  setFile(e.dataTransfer.files[0]);
});

// ----------------------------------------------------------------
// Analysis via SSE streaming
// ----------------------------------------------------------------
btnAnalyze.addEventListener('click', () => {
  if (!selectedFile) return;
  runAnalysis();
});

async function runAnalysis() {
  setLoading(true);
  clearResults();
  placeholder.style.display = 'none';

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const resp = await fetch('/api/analyze/stream', {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      showError(err.error || 'Request failed.');
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';
    let fullText  = '';
    let cursor    = null;
    let t0        = Date.now();

    // Show description section immediately for streaming effect
    descSection.style.display = '';
    descriptionBox.innerHTML  = '';
    cursor = document.createElement('span');
    cursor.className = 'cursor-blink';
    descriptionBox.appendChild(cursor);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();      // keep incomplete last line

      for (const line of lines) {
        if (line.startsWith('event: ')) continue;  // handled below via data pairing
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let parsed;
        try { parsed = JSON.parse(raw); } catch { continue; }

        // Determine event type from preceding line (use buffer trick or look back)
        // Since we skipped event lines, detect by payload shape
        if (Array.isArray(parsed)) {
          // CLIP results
          renderClipResults(parsed);
          clipSection.style.display = '';
          placeholder.style.display = 'none';
        } else if (parsed.token !== undefined) {
          // Streaming token
          fullText += parsed.token;
          if (cursor) cursor.remove();
          descriptionBox.textContent = fullText;
          cursor = document.createElement('span');
          cursor.className = 'cursor-blink';
          descriptionBox.appendChild(cursor);
        } else if (parsed.elapsed_ms !== undefined) {
          // Done
          if (cursor) cursor.remove();
          descriptionBox.textContent = fullText;
          elapsedMs.textContent      = `⏱ ${(parsed.elapsed_ms / 1000).toFixed(2)}s`;
          metaRow.style.display      = 'flex';
        } else if (parsed.error) {
          showError(parsed.error);
        }
      }
    }
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    setLoading(false);
  }
}

// ----------------------------------------------------------------
// Render CLIP label chips
// ----------------------------------------------------------------
function renderClipResults(results) {
  labelGrid.innerHTML = '';
  results.forEach((r, i) => {
    const chip = document.createElement('div');
    const cls  = i < 3 ? 'label-chip high' : i < 7 ? 'label-chip mid' : 'label-chip';
    chip.className = cls;
    chip.innerHTML = `${r.label} <span class="chip-pct">${r.confidence}%</span>`;
    labelGrid.appendChild(chip);
  });
}

// ----------------------------------------------------------------
// UI helpers
// ----------------------------------------------------------------
function setLoading(loading) {
  btnAnalyze.disabled       = loading;
  btnLabel.style.display    = loading ? 'none' : '';
  btnSpinner.style.display  = loading ? '' : 'none';
}

function showError(msg) {
  errorMsg.textContent     = msg;
  errorBox.style.display   = 'flex';
  placeholder.style.display = 'none';
}

// ----------------------------------------------------------------
// Copy description
// ----------------------------------------------------------------
btnCopy.addEventListener('click', () => {
  const text = descriptionBox.textContent.trim();
  navigator.clipboard.writeText(text).then(() => {
    btnCopy.textContent = '✅ Copied!';
    setTimeout(() => { btnCopy.textContent = '📋 Copy'; }, 2000);
  });
});

function copyCurl() {
  const code = document.getElementById('curlExample').textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.querySelector('.btn-copy-code');
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  });
}
