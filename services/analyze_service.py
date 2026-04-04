"""Single-image analyze API (BLIP/CLIP + Ollama description)."""
import time

from extensions import get_ollama, run_vision, vision_backend
from utils.image_utils import load_image_from_bytes


def analyze_image_to_json(image_bytes: bytes) -> dict:
    """Full pipeline → JSON-serializable dict. Raises RuntimeError from Ollama handler."""
    image = load_image_from_bytes(image_bytes)
    t0 = time.perf_counter()
    vision = run_vision(image)
    description = get_ollama().generate_description(
        clip_results=vision["clip_results"] or None,
        blip_caption=vision["blip_caption"],
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "vision_backend": vision_backend(),
        "blip_caption": vision["blip_caption"],
        "clip_results": vision["clip_results"],
        "description": description,
        "elapsed_ms": elapsed_ms,
    }
