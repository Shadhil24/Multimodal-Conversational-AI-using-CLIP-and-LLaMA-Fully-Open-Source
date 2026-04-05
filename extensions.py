"""
Shared model singletons and vision helpers (used by API services and FastAPI app).
"""
from config import Config
from model.blip_handler import BlipHandler
from model.clip_handler import CLIPHandler
from model.ollama_handler import OllamaHandler
from model.chat_ollama import ChatOllama

_clip: CLIPHandler | None = None
_blip: BlipHandler | None = None
_ollama: OllamaHandler | None = None
_chat_ollama: ChatOllama | None = None


def vision_backend() -> str:
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


def get_chat_ollama() -> ChatOllama:
    global _chat_ollama
    if _chat_ollama is None:
        _chat_ollama = ChatOllama()
    return _chat_ollama


def run_vision(image):
    """Run BLIP and/or CLIP depending on VISION_BACKEND."""
    backend = vision_backend()
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


def preload_vision_models() -> None:
    b = vision_backend()
    if b in ("blip", "both"):
        get_blip()
    if b in ("clip", "both"):
        get_clip()
