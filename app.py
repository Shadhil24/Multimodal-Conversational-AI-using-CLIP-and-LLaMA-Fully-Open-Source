import os
import io
import base64
import json
import time

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from model.clip_handler import CLIPHandler
from model.ollama_handler import OllamaHandler
from utils.image_utils import allowed_file, load_image_from_bytes, resize_for_preview

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Lazy-loaded singletons
_clip: CLIPHandler | None = None
_ollama: OllamaHandler | None = None


def get_clip() -> CLIPHandler:
    global _clip
    if _clip is None:
        _clip = CLIPHandler()
    return _clip


def get_ollama() -> OllamaHandler:
    global _ollama
    if _ollama is None:
        _ollama = OllamaHandler()
    return _ollama


# ---------------------------------------------------------------------------
# UI Route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def status():
    ollama_ok = get_ollama().is_available()
    return jsonify({
        "status": "ok",
        "ollama_available": ollama_ok,
        "ollama_model": Config.OLLAMA_MODEL,
        "clip_model": Config.CLIP_MODEL_NAME,
    })


# ---------------------------------------------------------------------------
# Core API: Analyse image → CLIP labels + Ollama description
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts a multipart/form-data POST with an image file.
    Returns JSON:
        {
            "clip_results": [ {"label": ..., "confidence": ...}, ... ],
            "description":  "<generated text>",
            "elapsed_ms":   <int>
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 415

    image_bytes = file.read()
    image = load_image_from_bytes(image_bytes)

    t0 = time.perf_counter()

    # Step 1 – CLIP zero-shot classification
    try:
        clip_results = get_clip().analyze_image(image)
    except Exception as e:
        return jsonify({"error": f"CLIP error: {e}"}), 500

    # Step 2 – Ollama description generation
    try:
        description = get_ollama().generate_description(clip_results)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return jsonify({
        "clip_results": clip_results,
        "description": description,
        "elapsed_ms": elapsed_ms,
    })


# ---------------------------------------------------------------------------
# Streaming API (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.route("/api/analyze/stream", methods=["POST"])
def analyze_stream():
    """
    Same as /api/analyze but streams the LLM response token by token
    using Server-Sent Events (SSE).

    SSE event types:
        - clip   : JSON array of CLIP results (sent once)
        - token  : one LLM token
        - done   : signals completion with elapsed_ms
        - error  : error message
    """
    if "image" not in request.files:
        def err():
            yield 'data: {"error": "No image provided"}\n\n'
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    file = request.files["image"]
    if not allowed_file(file.filename):
        def err():
            yield 'data: {"error": "Unsupported file type"}\n\n'
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    image_bytes = file.read()
    image = load_image_from_bytes(image_bytes)

    def generate():
        t0 = time.perf_counter()

        # CLIP step
        try:
            clip_results = get_clip().analyze_image(image)
        except Exception as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
            return

        yield f'event: clip\ndata: {json.dumps(clip_results)}\n\n'

        # Streaming LLM step
        try:
            for token in get_ollama().generate_stream(clip_results):
                yield f'event: token\ndata: {json.dumps({"token": token})}\n\n'
        except Exception as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
            return

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        yield f'event: done\ndata: {json.dumps({"elapsed_ms": elapsed_ms})}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ImageBot – CLIP + Ollama Image-to-Text")
    print(f"  CLIP model  : {Config.CLIP_MODEL_NAME}")
    print(f"  LLM model   : {Config.OLLAMA_MODEL}")
    print(f"  Ollama URL  : {Config.OLLAMA_BASE_URL}")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    # Pre-load CLIP on startup so the first request is fast
    get_clip()
    app.run(debug=True, host="0.0.0.0", port=5000)
