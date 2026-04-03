import json
import requests

from config import Config


def format_user_turn(user_text: str | None, blip_caption: str | None) -> str:
    """Build a single user message for Ollama from optional text + optional image caption."""
    t = (user_text or "").strip()
    if blip_caption and t:
        return (
            "[Note—rough machine read of their photo, for you only—don't repeat back unless they ask]\n"
            f"{blip_caption}\n\n"
            f"They texted:\n{t}\n\n"
            "Answer like their friend: straight to the point about what they asked. "
            "Don't analyze the picture, the wall, the lighting, or what this could work for "
            "unless that's what they're asking."
        )
    if blip_caption and not t:
        return (
            "[Note—rough machine read of their photo, for you only]\n"
            f"{blip_caption}\n\n"
            "They only sent a pic, no words. Reply like a friend would—short, genuine, maybe "
            "one casual reaction to what's probably in it. Not a formal paragraph-by-paragraph "
            "description unless it really fits the vibe."
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

    def _chat_options(self) -> dict:
        return {
            "temperature": Config.OLLAMA_CHAT_TEMPERATURE,
            "top_p": 0.93,
            "num_predict": Config.OLLAMA_CHAT_MAX_TOKENS,
        }

    def complete(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._chat_options(),
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
            "options": self._chat_options(),
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
