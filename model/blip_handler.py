import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from config import Config


class BlipHandler:
    """
    BLIP generates a free-form caption from pixels (not a fixed label list).

    This avoids CLIP-style errors like mapping two children to "a couple"
    when that phrase is in the candidate list but no better phrase exists.
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

    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_length=Config.BLIP_MAX_CAPTION_LEN,
                num_beams=4,
            )
        caption = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return caption.strip()
