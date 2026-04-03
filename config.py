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
    BLIP_MAX_CAPTION_LEN = int(os.getenv("BLIP_MAX_CAPTION_LEN", "140"))
    BLIP_MIN_CAPTION_LEN = int(os.getenv("BLIP_MIN_CAPTION_LEN", "18"))
    BLIP_NUM_BEAMS = int(os.getenv("BLIP_NUM_BEAMS", "5"))
    BLIP_REPETITION_PENALTY = float(os.getenv("BLIP_REPETITION_PENALTY", "1.22"))
    BLIP_LENGTH_PENALTY = float(os.getenv("BLIP_LENGTH_PENALTY", "1.12"))
    BLIP_USE_DUAL_CAPTION = os.getenv("BLIP_USE_DUAL_CAPTION", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # Second BLIP pass: conditional prefix (model completes the sentence)
    BLIP_CONDITIONAL_PREFIX = os.getenv(
        "BLIP_CONDITIONAL_PREFIX",
        "a detailed photograph showing",
    )

    # Ollama config
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    OLLAMA_TIMEOUT = 120  # seconds
    OLLAMA_DESCRIPTION_MAX_TOKENS = int(os.getenv("OLLAMA_DESCRIPTION_MAX_TOKENS", "420"))
    OLLAMA_CHAT_MAX_TOKENS = int(os.getenv("OLLAMA_CHAT_MAX_TOKENS", "600"))

    # Multi-turn chat (POST /api/chat)
    CHAT_MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "40"))
    CHAT_SYSTEM_PROMPT = os.getenv(
        "CHAT_SYSTEM_PROMPT",
        "You are a helpful assistant. Users may send text only, an image only, or both. "
        "When an 'Automatic image description' block appears, it is grounded context from a "
        "vision model: answer with specific, vivid detail the description supports—subjects, "
        "setting, objects, colors or lighting when implied, spatial layout when clear. Use "
        "'appears', 'likely', or 'possibly' when fine detail is uncertain. Do not invent "
        "people, objects, or actions that are not supported by the description or the user's "
        "words. For plain text-only chat, stay clear and reasonably concise unless they ask "
        "for depth.",
    )
