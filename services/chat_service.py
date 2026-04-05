"""Chat API business logic (Ollama + optional vision)."""
import json
from typing import Generator

from config import Config
from extensions import get_chat_ollama, run_vision
from model.chat_ollama import (
    build_chat_messages,
    format_user_turn,
    validate_history,
)
from utils.image_utils import allowed_file, load_image_from_bytes


def parse_history_json(history_raw: str) -> list[dict]:
    """Parse the `history` form field (JSON array of messages)."""
    try:
        return validate_history(json.loads(history_raw or "[]"))
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Invalid history JSON") from None


def run_chat_completion(
    *,
    history: list[dict],
    message: str,
    image_bytes: bytes | None,
    filename: str | None,
) -> tuple[str, str]:
    """
    Returns (reply, user_content). Raises on vision/ollama errors.
    """
    blip_caption: str | None = None
    if image_bytes and filename:
        if not allowed_file(filename):
            raise ValueError(f"Unsupported image type. Allowed: {Config.ALLOWED_EXTENSIONS}")
        vision = run_vision(load_image_from_bytes(image_bytes))
        blip_caption = vision.get("blip_caption")

    user_content = format_user_turn(message or None, blip_caption)
    if not user_content:
        raise ValueError("Send a non-empty message and/or an image.")

    ollama_messages = build_chat_messages(history, user_content)
    reply = get_chat_ollama().complete(ollama_messages)
    return reply.strip(), user_content


def stream_chat_tokens(ollama_messages: list[dict]) -> Generator[str, None, None]:
    yield from get_chat_ollama().stream_complete(ollama_messages)


def build_chat_stream_context(
    *,
    history: list[dict],
    message: str,
    image_bytes: bytes | None,
    filename: str | None,
) -> tuple[str, list[dict]]:
    """Returns (user_content, ollama_messages) or raises."""
    blip_caption: str | None = None
    if image_bytes and filename:
        if not allowed_file(filename):
            raise ValueError("Unsupported image type")
        vision = run_vision(load_image_from_bytes(image_bytes))
        blip_caption = vision.get("blip_caption")

    user_content = format_user_turn(message or None, blip_caption)
    if not user_content:
        raise ValueError("Send a message and/or an image.")

    ollama_messages = build_chat_messages(history, user_content)
    return user_content, ollama_messages
