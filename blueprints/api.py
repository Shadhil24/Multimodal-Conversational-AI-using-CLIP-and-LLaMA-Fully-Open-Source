"""
REST API blueprint — all JSON/SSE endpoints. UI pages call these via fetch().
"""
import json
import time

import requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from config import Config
from extensions import get_ollama, preload_vision_models, run_vision, vision_backend
from services import analyze_service, chat_service
from utils.image_utils import allowed_file, load_image_from_bytes

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("", methods=["GET"])
@api_bp.route("/", methods=["GET"])
def api_index():
    """Machine-readable list of endpoints (for clients and docs)."""
    base = request.url_root.rstrip("/")
    return jsonify({
        "service": "ImageBot",
        "base_url": f"{base}/api",
        "endpoints": [
            {
                "path": "/api",
                "method": "GET",
                "description": "This discovery document",
            },
            {
                "path": "/api/status",
                "method": "GET",
                "description": "Health and model configuration",
            },
            {
                "path": "/api/chat",
                "method": "POST",
                "content_type": "multipart/form-data",
                "fields": {
                    "history": "JSON array of {role, content} (user|assistant)",
                    "message": "optional user text",
                    "image": "optional image file",
                },
                "returns": {"reply": "string", "user_content": "string"},
            },
            {
                "path": "/api/chat/stream",
                "method": "POST",
                "content_type": "multipart/form-data",
                "same_as": "/api/chat",
                "returns": "Server-Sent Events: meta, token, done, error",
            },
            {
                "path": "/api/analyze",
                "method": "POST",
                "content_type": "multipart/form-data",
                "fields": {"image": "required image file"},
                "returns": "JSON with blip_caption, clip_results, description, elapsed_ms",
            },
            {
                "path": "/api/analyze/stream",
                "method": "POST",
                "content_type": "multipart/form-data",
                "fields": {"image": "required image file"},
                "returns": "SSE: blip, clip, token, done, error",
            },
        ],
    })


@api_bp.route("/status", methods=["GET"])
def status():
    ollama_ok = get_ollama().is_available()
    vb = vision_backend()
    return jsonify({
        "status": "ok",
        "vision_backend": vb,
        "blip_model": Config.BLIP_MODEL_NAME if vb in ("blip", "both") else None,
        "clip_model": Config.CLIP_MODEL_NAME if vb in ("clip", "both") else None,
        "ollama_available": ollama_ok,
        "ollama_model": Config.OLLAMA_MODEL,
    })


@api_bp.route("/chat", methods=["POST"])
def api_chat():
    try:
        history, message, image_file = chat_service.parse_chat_multipart(
            request.form, request.files
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    image_bytes: bytes | None = None
    filename: str | None = None
    if image_file and image_file.filename:
        filename = image_file.filename
        image_bytes = image_file.read()

    try:
        reply, user_content = chat_service.run_chat_completion(
            history=history,
            message=message,
            image_bytes=image_bytes,
            filename=filename,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Ollama error: {e}"}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Cannot reach Ollama: {e}"}), 503
    except Exception as e:
        return jsonify({"error": f"Vision or model error: {e}"}), 500

    return jsonify({"reply": reply, "user_content": user_content})


@api_bp.route("/chat/stream", methods=["POST"])
def api_chat_stream():
    try:
        history, message, image_file = chat_service.parse_chat_multipart(
            request.form, request.files
        )
    except ValueError as e:

        def err1():
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

        return Response(stream_with_context(err1()), mimetype="text/event-stream")

    image_bytes: bytes | None = None
    filename: str | None = None
    if image_file and image_file.filename:
        filename = image_file.filename
        image_bytes = image_file.read()

    try:
        user_content, ollama_messages = chat_service.build_chat_stream_context(
            history=history,
            message=message,
            image_bytes=image_bytes,
            filename=filename,
        )
    except ValueError as e:

        def err2():
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

        return Response(stream_with_context(err2()), mimetype="text/event-stream")
    except Exception as e:

        def err2b():
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

        return Response(stream_with_context(err2b()), mimetype="text/event-stream")

    def generate():
        yield f'event: meta\ndata: {json.dumps({"user_content": user_content})}\n\n'
        try:
            for piece in chat_service.stream_chat_tokens(ollama_messages):
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


@api_bp.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 415

    try:
        return jsonify(analyze_service.analyze_image_to_json(file.read()))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Vision model error: {e}"}), 500


@api_bp.route("/analyze/stream", methods=["POST"])
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


def register_api(app):
    app.register_blueprint(api_bp)
