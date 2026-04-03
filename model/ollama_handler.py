import json
import requests
from config import Config


def _filter_labels(clip_results: list[dict]) -> list[dict]:
    """
    Remove noisy low-confidence labels before sending to the LLM.

    With ~100 candidate labels, the softmax noise floor is ≈1%.
    We keep only labels that are above CLIP_MIN_CONFIDENCE (default 10%),
    capped at CLIP_TOP_K_TO_LLM entries.  If nothing passes the threshold
    we fall back to just the single top label so the LLM always has something.
    """
    filtered = [r for r in clip_results if r["confidence"] >= Config.CLIP_MIN_CONFIDENCE]
    filtered = filtered[: Config.CLIP_TOP_K_TO_LLM]
    if not filtered:
        filtered = clip_results[:1]   # fallback: keep the single best label
    return filtered


def _build_prompt(clip_results: list[dict]) -> str:
    filtered = _filter_labels(clip_results)
    confidence_lines = "\n".join(
        f"  - {r['label']} ({r['confidence']}%)" for r in filtered
    )

    # Tell the LLM exactly how many reliable signals we have so it doesn't
    # invent extra elements that aren't in the image.
    count = len(filtered)
    focus_note = (
        "Only ONE element was detected with high confidence."
        if count == 1
        else f"Only {count} elements were detected with high confidence."
    )

    return (
        "You are an expert image analyst. "
        f"{focus_note} "
        "Based ONLY on the detected elements listed below, write a single "
        "concise and accurate sentence or two describing what this image most "
        "likely shows. Do NOT invent or add any objects, animals, or scenes "
        "that are not in the list. Stick strictly to what is detected.\n\n"
        f"High-confidence detected elements:\n{confidence_lines}\n\n"
        "Image description:"
    )


class OllamaHandler:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT

    def is_available(self) -> bool:
        """Check if Ollama service is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate_description(self, clip_results: list[dict]) -> str:
        """
        Build a filtered prompt from CLIP results and ask the LLM for a
        grounded, accurate image description.
        """
        prompt = _build_prompt(clip_results)

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

    def generate_stream(self, clip_results: list[dict]):
        """
        Generator that yields tokens from Ollama's streaming API.
        Uses the same filtered, grounded prompt as generate_description.
        """
        prompt = _build_prompt(clip_results)

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
