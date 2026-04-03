import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "imagebot-secret-key")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

    # CLIP model
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CLIP_TOP_K = 15       # Maximum labels to retrieve from CLIP
    CLIP_TOP_K_TO_LLM = 5 # Max labels actually sent to the LLM
    # Only labels above this threshold are sent to Ollama.
    # With ~100 candidates, uniform chance ≈ 1%.  10% = 10× above noise.
    CLIP_MIN_CONFIDENCE = 10.0

    # Vision backend: "blip" = image captioning (recommended), "clip" = label list only,
    # "both" = BLIP caption + CLIP tag chips (two models in memory).
    VISION_BACKEND = os.getenv("VISION_BACKEND", "blip").strip().lower()

    # BLIP image captioning (Salesforce/blip-image-captioning-base ≈ 990M params)
    BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
    BLIP_MAX_CAPTION_LEN = 75

    # Ollama config
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    OLLAMA_TIMEOUT = 120  # seconds
