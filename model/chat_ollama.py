import json
import requests

from config import Config


def format_user_turn(user_text: str | None, blip_caption: str | None) -> str:
    """Build a single user message for Ollama from optional text + optional image caption."""
    t = (user_text or "").strip()
    if blip_caption and t:
        return (
            "Automatic image description (from vision model):\n"
            f"{blip_caption}\n\n"
            f"User message:\n{t}"
        )
    if blip_caption and not t:
        return (
            "Automatic image description (from vision model):\n"
            f"{blip_caption}\n\n"
            "The user did not type a separate message. Briefly describe what is shown "
            "and invite them to ask follow-up questions about the image."
        )
    if not blip_caption and t:
        return t
    return ""


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def validate_history(history: list | None) -> list[dict]:
    if not history:
        return []
    out: list[dict] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str):
            continue
        c = content.strip()
        if not c:
            continue
        if len(c) > 12000:
            c = c[:12000] + "…"
        out.append({"role": role, "content": c})
    return out


def build_chat_messages(history: list[dict], user_content: str) -> list[dict]:
    sys = Config.CHAT_SYSTEM_PROMPT.strip()
    messages: list[dict] = [{"role": "system", "content": sys}]
    messages.extend(trim_history(history, Config.CHAT_MAX_HISTORY_MESSAGES))
    messages.append({"role": "user", "content": user_content.strip()})
    return messages


class ChatOllama:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

    def complete(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.6, "top_p": 0.9},
        }
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()

    def stream_complete(self, messages: list[dict]):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.6, "top_p": 0.9},
        }
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("done"):
                    break
                msg = chunk.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield piece
