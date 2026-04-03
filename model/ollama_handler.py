import json
import requests
from config import Config


def _filter_labels(clip_results: list[dict]) -> list[dict]:
    filtered = [r for r in clip_results if r["confidence"] >= Config.CLIP_MIN_CONFIDENCE]
    filtered = filtered[: Config.CLIP_TOP_K_TO_LLM]
    if not filtered and clip_results:
        filtered = clip_results[:1]
    return filtered


def _build_prompt(*, clip_results: list[dict] | None, blip_caption: str | None) -> str:
    """
    If blip_caption is set, it is the primary visual signal (open vocabulary).
    Optional clip_results add coarse tags; Ollama should not override BLIP with wrong tags.
    """
    parts = ["You are an expert image analyst."]

    if blip_caption:
        parts.append(
            "An image captioning model produced this draft description.\n"
            f"---\n{blip_caption}\n---\n"
            "Write one or two clear, natural sentences for a general audience. "
            "Stay faithful to the draft: do not add people, objects, or relationships "
            "that are not supported by it, and do not contradict it.\n"
        )

    if clip_results:
        filtered = _filter_labels(clip_results)
        confidence_lines = "\n".join(
            f"  - {r['label']} ({r['confidence']}%)" for r in filtered
        )
        count = len(filtered)
        focus_note = (
            "Only ONE coarse tag exceeded the confidence threshold."
            if count == 1
            else f"Only {count} coarse tags exceeded the confidence threshold."
        )
        if blip_caption:
            parts.append(
                f"Optional extra tags from a separate classifier ({focus_note}). "
                "Use them only if they agree with the draft above; ignore conflicting tags.\n"
                f"Tags:\n{confidence_lines}\n"
            )
        else:
            parts.append(
                f"{focus_note} Based ONLY on the tags below, write one or two concise "
                "sentences describing what the image most likely shows. Do NOT invent "
                "objects or scenes not implied by the tags.\n\n"
                f"Tags:\n{confidence_lines}\n"
            )

    if not blip_caption and not clip_results:
        parts.append("No visual signal was provided.")

    parts.append("\nImage description:")
    return "".join(parts)


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

    def generate_description(
        self,
        *,
        clip_results: list[dict] | None = None,
        blip_caption: str | None = None,
    ) -> str:
        prompt = _build_prompt(
            clip_results=clip_results or [],
            blip_caption=blip_caption,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "num_predict": 200,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
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
        prompt = _build_prompt(
            clip_results=clip_results or [],
            blip_caption=blip_caption,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.5, "top_p": 0.9, "num_predict": 200},
        }

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
