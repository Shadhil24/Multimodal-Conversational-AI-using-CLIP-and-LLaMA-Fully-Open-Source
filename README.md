# ImageBot – AI Image-to-Text

A Flask web application that combines **OpenAI CLIP** for visual understanding and **Llama 3.2 (via Ollama)** to generate natural-language descriptions of any uploaded image.

---

## How it Works

```
User uploads image
      │
      ▼
 CLIP Model (ViT-B/32)
 Zero-shot classification
 against 100+ candidate labels
      │
      ▼
 Top-K labels + confidence scores
      │
      ▼
 Ollama  ──►  llama3.2:1b
 Generates a flowing paragraph
 description of the image
      │
      ▼
 Streamed back to the browser
```

---

## Project Structure

```
imagebot/
├── app.py                  # Flask application + API routes
├── config.py               # Configuration (models, paths, limits)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── model/
│   ├── clip_handler.py     # CLIP zero-shot classification
│   └── ollama_handler.py   # Ollama LLM integration
├── utils/
│   └── image_utils.py      # Image loading & validation helpers
├── templates/
│   └── index.html          # Flask UI (dark-mode, drag-and-drop)
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| CUDA (optional) | any | GPU acceleration for CLIP |

---

## Setup

### 1. Clone & enter the project

```bash
git clone https://github.com/YOUR_USERNAME/imagebot.git
cd imagebot
```

### 2. Create & activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch + CLIP weights (~350 MB) will be downloaded on first run.

### 4. Set up Ollama

```bash
# Install Ollama from https://ollama.com/download
# Then pull the model
ollama pull llama3.2:1b
```

### 5. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if you need custom values
```

### 6. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## REST API

### `GET /api/status`
Returns service health and model info.

```json
{
  "status": "ok",
  "ollama_available": true,
  "ollama_model": "llama3.2:1b",
  "clip_model": "openai/clip-vit-base-patch32"
}
```

---

### `POST /api/analyze`
Upload an image and receive the full result as JSON.

**Request** – `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Image file (PNG, JPG, JPEG, GIF, WEBP, BMP) |

**Response**
```json
{
  "clip_results": [
    { "label": "a dog", "confidence": 32.5 },
    { "label": "an outdoor scene", "confidence": 18.1 },
    ...
  ],
  "description": "The image shows a golden retriever playing in...",
  "elapsed_ms": 2350
}
```

**curl example**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "image=@/path/to/your/image.jpg"
```

---

### `POST /api/analyze/stream`
Same as above but streams the LLM response token-by-token using **Server-Sent Events (SSE)**.

**SSE Events**

| Event | Payload | Description |
|-------|---------|-------------|
| `clip` | JSON array | CLIP results (sent once) |
| `token` | `{"token": "..."}` | LLM token |
| `done` | `{"elapsed_ms": N}` | Completion signal |
| `error` | `{"error": "..."}` | Error message |

---

## Tech Stack

- **[OpenAI CLIP](https://github.com/openai/CLIP)** via HuggingFace `transformers`
- **[Ollama](https://ollama.com)** – local LLM inference
- **[Llama 3.2:1b](https://ollama.com/library/llama3.2)** – text generation
- **[Flask](https://flask.palletsprojects.com/)** – web framework
- **PyTorch** – model inference

---

## License

MIT
