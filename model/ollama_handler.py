import re
import requests
from config import Config

# Small chat models often false-trigger on words like "relationships" in instructions.
# Keep prompts in “alt-text / grammar fix” framing; add system + refusal fallback.

OLLAMA_SYSTEM_PROMPT = (
    "You write rich, precise image descriptions for documentation and accessibility. "
    "Input is neutral output from computer vision about an ordinary photograph. "
    "Reply with only the final description text—no preamble. "
    "Expand with concrete, visible detail the draft supports (subjects, setting, "
    "notable objects, colors, lighting, spatial layout). Use cautious wording "
    "(appears, likely, seems) when the draft is vague. "
    "Do not refuse harmless content. Do not invent people, objects, or actions "
    "not clearly supported by the draft."
)


def _filter_labels(clip_results: list[dict]) -> list[dict]:
    filtered = [r for r in clip_results if r["confidence"] >= Config.CLIP_MIN_CONFIDENCE]
    filtered = filtered[: Config.CLIP_TOP_K_TO_LLM]
    if not filtered and clip_results:
        filtered = clip_results[:1]
    return filtered


def _looks_like_model_refusal(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.lower()
    needles = (
        "cannot write content",
        "can't write content",
        "i cannot write",
        "i can't write",
        "sexual relationship",
        "adults and minors",
        "i'm not able to",
        "i am not able to",
        "as an ai",
        "can i help you with something else",
        "i apologize, but i cannot",
        "i apologize, but i can't",
    )
    return any(n in t for n in needles)


def _build_user_prompt(*, clip_results: list[dict] | None, blip_caption: str | None) -> str:
    """
    User message only (system holds role). Avoid “relationship” wording — it trips safety filters.
    """
    parts: list[str] = []

    if blip_caption:
        parts.append(
            "Draft description from the vision model (raw; may be terse):\n"
            f'"""{blip_caption}"""\n\n'
            "Turn this into a vivid, well-organized description of 3–6 sentences. "
            "Lead with the main subject(s) and setting, then add specifics the draft "
            "supports: objects, clothing or appearance if mentioned, colors or lighting "
            "if implied, foreground/background when inferable. "
            "Do not add characters, props, or actions that are not grounded in the draft. "
            "If the draft is short, deepen only by careful elaboration of what it already states."
        )

    if clip_results:
        filtered = _filter_labels(clip_results)
        confidence_lines = "\n".join(
            f"  - {r['label']} ({r['confidence']}%)" for r in filtered
        )
        count = len(filtered)
        focus_note = (
            "One optional classifier tag passed the confidence threshold."
            if count == 1
            else f"{count} optional classifier tags passed the confidence threshold."
        )
        if blip_caption:
            parts.append(
                f"{focus_note} Use these tags only if they match the draft; otherwise ignore:\n"
                f"{confidence_lines}"
            )
        else:
            parts.append(
                f"{focus_note} Write one or two sentences describing the image using ONLY these tags. "
                "Do not invent objects or scenes not hinted at by the tags.\n"
                f"{confidence_lines}"
            )

    if not blip_caption and not clip_results:
        parts.append("No vision output was provided.")

    parts.append("\nPolished description:")
    return "\n\n".join(parts)


class OllamaHandler:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _call_generate(self, *, prompt: str, stream: bool):
        payload = {
            "model": self.model,
            "system": OLLAMA_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.42,
                "top_p": 0.92,
                "num_predict": Config.OLLAMA_DESCRIPTION_MAX_TOKENS,
            },
        }
        return requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=stream,
            timeout=self.timeout,
        )

    def _finalize_text(self, raw: str, *, blip_caption: str | None) -> tuple[str, bool]:
        """
        Returns (text, used_fallback). If the model refused, return vision caption as description.
        """
        text = (raw or "").strip()
        if _looks_like_model_refusal(text) and blip_caption:
            return blip_caption.strip(), True
        return text, False

    def generate_description(
        self,
        *,
        clip_results: list[dict] | None = None,
        blip_caption: str | None = None,
    ) -> str:
        prompt = _build_user_prompt(
            clip_results=clip_results or [],
            blip_caption=blip_caption,
        )
        try:
            response = self._call_generate(prompt=prompt, stream=False)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")
            text, fb = self._finalize_text(raw, blip_caption=blip_caption)
            if fb:
                print("[Ollama] Refusal or empty reply; using vision caption as description.")
            return text
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running "
                f"at {self.base_url} and the model '{self.model}' is pulled."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out. Try again.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_stream(
        self,
        *,
        clip_results: list[dict] | None = None,
        blip_caption: str | None = None,
    ):
        """
        Use non-streaming inference first (same as JSON API), then emit word chunks so
        refusals can be replaced by the BLIP caption before any token hits the client.
        """
        prompt = _build_user_prompt(
            clip_results=clip_results or [],
            blip_caption=blip_caption,
        )
        try:
            response = self._call_generate(prompt=prompt, stream=False)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")
            text, fb = self._finalize_text(raw, blip_caption=blip_caption)
            if fb:
                print("[Ollama] Refusal or empty reply; using vision caption as description.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure Ollama is running "
                f"at {self.base_url} and the model '{self.model}' is pulled."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out. Try again.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

        for m in re.finditer(r"\S+\s*", text):
            yield m.group(0)
