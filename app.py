import json
import time

import requests
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from model.blip_handler import BlipHandler
from model.chat_ollama import (
    ChatOllama,
    build_chat_messages,
    format_user_turn,
    validate_history,
)
from model.clip_handler import CLIPHandler
from model.ollama_handler import OllamaHandler
from utils.image_utils import allowed_file, load_image_from_bytes

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

_clip: CLIPHandler | None = None
_blip: BlipHandler | None = None
_ollama: OllamaHandler | None = None
_chat_ollama: ChatOllama | None = None


def get_chat_ollama() -> ChatOllama:
    global _chat_ollama
    if _chat_ollama is None:
        _chat_ollama = ChatOllama()
    return _chat_ollama


def _parse_chat_request():
    raw = request.form.get("history", "[]")
    try:
        history = validate_history(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Invalid history JSON") from None
    message = request.form.get("message", "") or ""
    image_file = request.files.get("image")
    return history, message, image_file


def _vision_backend() -> str:
    b = (Config.VISION_BACKEND or "blip").strip().lower()
    if b not in ("blip", "clip", "both"):
        return "blip"
    return b


def get_clip() -> CLIPHandler:
    global _clip
    if _clip is None:
        _clip = CLIPHandler()
    return _clip


def get_blip() -> BlipHandler:
    global _blip
    if _blip is None:
        _blip = BlipHandler()
    return _blip


def get_ollama() -> OllamaHandler:
    global _ollama
    if _ollama is None:
        _ollama = OllamaHandler()
    return _ollama


def run_vision(image):
    """
    Run BLIP and/or CLIP depending on VISION_BACKEND.
    Returns dict: clip_results (list), blip_caption (str | None)
    """
    backend = _vision_backend()
    clip_results: list = []
    blip_caption: str | None = None

    if backend in ("clip", "both"):
        clip_results = get_clip().analyze_image(image)
    if backend in ("blip", "both"):
        blip_caption = get_blip().caption(image)

    return {
        "clip_results": clip_results,
        "blip_caption": blip_caption,
    }


def preload_vision_models():
    """Load models at startup based on configured backend."""
    b = _vision_backend()
    if b in ("blip", "both"):
        get_blip()
    if b in ("clip", "both"):
        get_clip()


# ---------------------------------------------------------------------------
# UI Route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/legacy")
def legacy():
    return render_template("legacy.html")


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def status():
    ollama_ok = get_ollama().is_available()
    vb = _vision_backend()
    return jsonify({
        "status": "ok",
        "vision_backend": vb,
        "blip_model": Config.BLIP_MODEL_NAME if vb in ("blip", "both") else None,
        "clip_model": Config.CLIP_MODEL_NAME if vb in ("clip", "both") else None,
        "ollama_available": ollama_ok,
        "ollama_model": Config.OLLAMA_MODEL,
    })


# ---------------------------------------------------------------------------
# Chat API (text, image, or both — multipart)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        history, message, image_file = _parse_chat_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    blip_caption: str | None = None
    if image_file and image_file.filename:
        if not allowed_file(image_file.filename):
            return jsonify({"error": f"Unsupported image type. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 415
        try:
            image_bytes = image_file.read()
            vision = run_vision(load_image_from_bytes(image_bytes))
            blip_caption = vision.get("blip_caption")
        except Exception as e:
            return jsonify({"error": f"Vision error: {e}"}), 500

    user_content = format_user_turn(message or None, blip_caption)
    if not user_content:
        return jsonify({"error": "Send a non-empty message and/or an image."}), 400

    ollama_messages = build_chat_messages(history, user_content)
    try:
        reply = get_chat_ollama().complete(ollama_messages)
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Ollama error: {e}"}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Cannot reach Ollama: {e}"}), 503

    return jsonify({"reply": reply, "user_content": user_content})


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    try:
        history, message, image_file = _parse_chat_request()
    except ValueError as e:

        def err1():
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

        return Response(stream_with_context(err1()), mimetype="text/event-stream")

    blip_caption: str | None = None
    if image_file and image_file.filename:
        if not allowed_file(image_file.filename):

            def err2():
                yield f'event: error\ndata: {json.dumps({"error": "Unsupported image type"})}\n\n'

            return Response(stream_with_context(err2()), mimetype="text/event-stream")
        try:
            image_bytes = image_file.read()
            vision = run_vision(load_image_from_bytes(image_bytes))
            blip_caption = vision.get("blip_caption")
        except Exception as e:

            def err3():
                yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

            return Response(stream_with_context(err3()), mimetype="text/event-stream")

    user_content = format_user_turn(message or None, blip_caption)
    if not user_content:

        def err4():
            yield f'event: error\ndata: {json.dumps({"error": "Send a message and/or an image."})}\n\n'

        return Response(stream_with_context(err4()), mimetype="text/event-stream")

    ollama_messages = build_chat_messages(history, user_content)

    def generate():
        yield f'event: meta\ndata: {json.dumps({"user_content": user_content})}\n\n'
        try:
            for piece in get_chat_ollama().stream_complete(ollama_messages):
                yield f'event: token\ndata: {json.dumps({"token": piece})}\n\n'
        except requests.exceptions.RequestException as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
            return
        yield 'event: done\ndata: {}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
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

    try:
        vision = run_vision(image)
    except Exception as e:
        return jsonify({"error": f"Vision model error: {e}"}), 500

    try:
        description = get_ollama().generate_description(
            clip_results=vision["clip_results"] or None,
            blip_caption=vision["blip_caption"],
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return jsonify({
        "vision_backend": _vision_backend(),
        "blip_caption": vision["blip_caption"],
        "clip_results": vision["clip_results"],
        "description": description,
        "elapsed_ms": elapsed_ms,
    })


# ---------------------------------------------------------------------------
# Streaming API (SSE)
# ---------------------------------------------------------------------------

@app.route("/api/analyze/stream", methods=["POST"])
def analyze_stream():
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

        try:
            vision = run_vision(image)
        except Exception as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
            return

        if vision["blip_caption"]:
            yield f'event: blip\ndata: {json.dumps({"caption": vision["blip_caption"]})}\n\n'

        if vision["clip_results"]:
            yield f'event: clip\ndata: {json.dumps(vision["clip_results"])}\n\n'

        try:
            for token in get_ollama().generate_stream(
                clip_results=vision["clip_results"] or None,
                blip_caption=vision["blip_caption"],
            ):
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
    vb = _vision_backend()
    print("=" * 60)
    print("  ImageBot – Vision + Ollama")
    print(f"  Vision mode : {vb}")
    if vb in ("blip", "both"):
        print(f"  BLIP model  : {Config.BLIP_MODEL_NAME}")
    if vb in ("clip", "both"):
        print(f"  CLIP model  : {Config.CLIP_MODEL_NAME}")
    print(f"  LLM model   : {Config.OLLAMA_MODEL}")
    print(f"  Ollama URL  : {Config.OLLAMA_BASE_URL}")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    preload_vision_models()
    app.run(debug=True, host="0.0.0.0", port=5000)
