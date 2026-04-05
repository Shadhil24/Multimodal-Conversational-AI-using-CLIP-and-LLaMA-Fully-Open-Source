"""
ImageBot — FastAPI application.

Serves the chat/legacy HTML UI, static assets, and REST + SSE API under /api/*.
Run: uvicorn main:app --host 0.0.0.0 --port 5000 --reload
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import Config
from extensions import get_ollama, preload_vision_models, run_vision, vision_backend
from services import analyze_service, chat_service
from utils.image_utils import allowed_file, load_image_from_bytes

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    preload_vision_models()
    yield


app = FastAPI(title="ImageBot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

api = APIRouter(prefix="/api", tags=["api"])


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def page_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/legacy", response_class=HTMLResponse)
async def page_legacy(request: Request):
    return templates.TemplateResponse("legacy.html", {"request": request})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@api.get("")
@api.get("/")
def api_index(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "service": "ImageBot",
        "base_url": f"{base}/api",
        "endpoints": [
            {"path": "/api", "method": "GET", "description": "This discovery document"},
            {"path": "/api/status", "method": "GET", "description": "Health and model configuration"},
            {
                "path": "/api/chat",
                "method": "POST",
                "content_type": "multipart/form-data",
                "fields": {
                    "history": "JSON array of {role, content}",
                    "message": "optional user text",
                    "image": "optional image file",
                },
            },
            {"path": "/api/chat/stream", "method": "POST", "description": "SSE stream"},
            {"path": "/api/analyze", "method": "POST", "fields": {"image": "required"}},
            {"path": "/api/analyze/stream", "method": "POST", "description": "SSE"},
        ],
    }


@api.get("/status")
def api_status():
    ollama_ok = get_ollama().is_available()
    vb = vision_backend()
    return {
        "status": "ok",
        "vision_backend": vb,
        "blip_model": Config.BLIP_MODEL_NAME if vb in ("blip", "both") else None,
        "clip_model": Config.CLIP_MODEL_NAME if vb in ("clip", "both") else None,
        "ollama_available": ollama_ok,
        "ollama_model": Config.OLLAMA_MODEL,
    }


@api.post("/chat")
async def api_chat(
    history: str = Form("[]"),
    message: str = Form(""),
    image: UploadFile | None = File(None),
):
    try:
        h = chat_service.parse_history_json(history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    image_bytes: bytes | None = None
    filename: str | None = None
    if image is not None and image.filename:
        filename = image.filename
        image_bytes = await image.read()

    try:
        reply, user_content = chat_service.run_chat_completion(
            history=h,
            message=message,
            image_bytes=image_bytes,
            filename=filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision or model error: {e}") from e

    return {"reply": reply, "user_content": user_content}


def _sse_error(msg: str):
    yield f'event: error\ndata: {json.dumps({"error": msg})}\n\n'


@api.post("/chat/stream")
async def api_chat_stream(
    history: str = Form("[]"),
    message: str = Form(""),
    image: UploadFile | None = File(None),
):
    try:
        h = chat_service.parse_history_json(history)
    except ValueError as e:

        def err():
            yield from _sse_error(str(e))

        return StreamingResponse(err(), media_type="text/event-stream")

    image_bytes: bytes | None = None
    filename: str | None = None
    if image is not None and image.filename:
        filename = image.filename
        image_bytes = await image.read()

    try:
        user_content, ollama_messages = chat_service.build_chat_stream_context(
            history=h,
            message=message,
            image_bytes=image_bytes,
            filename=filename,
        )
    except ValueError as e:

        def err2():
            yield from _sse_error(str(e))

        return StreamingResponse(err2(), media_type="text/event-stream")
    except Exception as e:

        def err3():
            yield from _sse_error(str(e))

        return StreamingResponse(err3(), media_type="text/event-stream")

    def generate():
        yield f'event: meta\ndata: {json.dumps({"user_content": user_content})}\n\n'
        try:
            for piece in chat_service.stream_chat_tokens(ollama_messages):
                yield f'event: token\ndata: {json.dumps({"token": piece})}\n\n'
        except requests.exceptions.RequestException as e:
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api.post("/analyze")
async def api_analyze(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")
    if not allowed_file(image.filename):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}",
        )
    try:
        data = analyze_service.analyze_image_to_json(await image.read())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision model error: {e}") from e
    return data


@api.post("/analyze/stream")
async def api_analyze_stream(image: UploadFile = File(...)):
    if not image.filename:

        def e1():
            yield 'data: {"error": "No image provided"}\n\n'

        return StreamingResponse(e1(), media_type="text/event-stream")

    if not allowed_file(image.filename):

        def e2():
            yield 'data: {"error": "Unsupported file type"}\n\n'

        return StreamingResponse(e2(), media_type="text/event-stream")

    image_bytes = await image.read()
    pil_image = load_image_from_bytes(image_bytes)

    def generate():
        t0 = time.perf_counter()
        try:
            vision = run_vision(pil_image)
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

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


app.include_router(api)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _error_detail(exc: HTTPException) -> str:
    d = exc.detail
    if isinstance(d, str):
        return d
    return json.dumps(d)


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _error_detail(exc)},
        )
    raise exc


if __name__ == "__main__":
    import uvicorn

    vb = vision_backend()
    print("=" * 60)
    print("  ImageBot – FastAPI + /api")
    print(f"  Vision mode : {vb}")
    if vb in ("blip", "both"):
        print(f"  BLIP model  : {Config.BLIP_MODEL_NAME}")
    if vb in ("clip", "both"):
        print(f"  CLIP model  : {Config.CLIP_MODEL_NAME}")
    print(f"  LLM model   : {Config.OLLAMA_MODEL}")
    print(f"  Ollama URL  : {Config.OLLAMA_BASE_URL}")
    print("  API docs    : http://127.0.0.1:5000/docs")
    print("  Chat UI     : http://127.0.0.1:5000/")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
