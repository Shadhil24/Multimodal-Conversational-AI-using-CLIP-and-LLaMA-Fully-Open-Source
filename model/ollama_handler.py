import requests
from config import Config


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
        Build a prompt from CLIP results and ask the LLM to generate
        a natural-language image description.
        """
        top_labels = [r["label"] for r in clip_results]
        confidence_lines = "\n".join(
            f"  - {r['label']} ({r['confidence']}%)" for r in clip_results
        )

        prompt = (
            "You are an expert image analyst. Based on the following visual elements "
            "detected in an image (ranked by confidence), write a single coherent, "
            "vivid, and natural paragraph that describes what the image likely shows. "
            "Do NOT list the items—write a flowing description as if you are looking "
            "at the image yourself.\n\n"
            f"Detected elements:\n{confidence_lines}\n\n"
            "Image description:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 300,
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
        """
        top_labels = [r["label"] for r in clip_results]
        confidence_lines = "\n".join(
            f"  - {r['label']} ({r['confidence']}%)" for r in clip_results
        )

        prompt = (
            "You are an expert image analyst. Based on the following visual elements "
            "detected in an image (ranked by confidence), write a single coherent, "
            "vivid, and natural paragraph that describes what the image likely shows. "
            "Do NOT list the items—write a flowing description.\n\n"
            f"Detected elements:\n{confidence_lines}\n\n"
            "Image description:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 300},
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
                    import json
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
