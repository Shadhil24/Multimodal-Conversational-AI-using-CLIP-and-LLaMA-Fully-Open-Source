import os
from PIL import Image
from config import Config


def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in ALLOWED_EXTENSIONS."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def load_image(path: str) -> Image.Image:
    """Open an image from disk and convert to RGB."""
    return Image.open(path).convert("RGB")


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load a PIL image from raw bytes."""
    import io
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_for_preview(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Downscale an image so its longest side is at most max_size pixels."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
