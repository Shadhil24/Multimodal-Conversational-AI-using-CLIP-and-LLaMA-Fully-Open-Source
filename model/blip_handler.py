import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from config import Config


def sanitize_blip_caption(text: str) -> str:
    """BLIP sometimes emits small word glitches; clean before showing / sending to an LLM."""
    s = " ".join((text or "").split())
    for bad, good in (
        (" each other men", " each other"),
        (" next to each other men", " next to each other"),
    ):
        s = s.replace(bad, good)
    parts = s.split()
    out: list[str] = []
    for w in parts:
        if out and w.lower() == out[-1].lower():
            continue
        out.append(w)
    return " ".join(out).strip()


def _merge_caption_passes(primary: str, secondary: str) -> str:
    """Combine two BLIP passes when they add complementary detail."""
    a = sanitize_blip_caption(primary)
    b = sanitize_blip_caption(secondary)
    if not b or a.lower() == b.lower():
        return a
    if not a:
        return b
    if a.lower() in b.lower() and len(b) >= len(a):
        return b
    if b.lower() in a.lower() and len(a) >= len(b):
        return a
    wa, wb = set(a.lower().split()), set(b.lower().split())
    inter = len(wa & wb)
    denom = max(min(len(wa), len(wb)), 1)
    if inter / denom > 0.82:
        return a if len(a) >= len(b) else b
    return f"{a.rstrip('.')}. {b}"


class BlipHandler:
    """
    BLIP generates image captions. We use beam search + optional dual pass (unconditional +
    conditional prefix) for richer, more specific phrases before the LLM expands them.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        name = Config.BLIP_MODEL_NAME
        print(f"[BLIP] Loading '{name}' on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name)
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        print("[BLIP] Model loaded successfully.")

    def _generate_raw(self, image: Image.Image, conditional_prefix: str | None) -> str:
        if conditional_prefix:
            prefix = conditional_prefix.strip()
            if prefix and not prefix.endswith(" "):
                prefix = f"{prefix} "
            inputs = self.processor(
                images=image,
                text=prefix,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        gen_kw: dict = {
            "max_length": Config.BLIP_MAX_CAPTION_LEN,
            "num_beams": Config.BLIP_NUM_BEAMS,
            "repetition_penalty": Config.BLIP_REPETITION_PENALTY,
            "length_penalty": Config.BLIP_LENGTH_PENALTY,
        }
        if Config.BLIP_MIN_CAPTION_LEN > 0:
            gen_kw["min_length"] = Config.BLIP_MIN_CAPTION_LEN

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kw)
        return self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

    def caption(self, image: Image.Image) -> str:
        cap1 = self._generate_raw(image, conditional_prefix=None)
        cap1 = sanitize_blip_caption(cap1)

        if not Config.BLIP_USE_DUAL_CAPTION:
            return cap1

        prefix = (Config.BLIP_CONDITIONAL_PREFIX or "").strip()
        if not prefix:
            return cap1

        cap2 = self._generate_raw(image, conditional_prefix=prefix)
        merged = _merge_caption_passes(cap1, cap2)
        return sanitize_blip_caption(merged)
