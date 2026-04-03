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
    CLIP_TOP_K = 15
    CLIP_TOP_K_TO_LLM = 5
    CLIP_MIN_CONFIDENCE = 10.0

    VISION_BACKEND = os.getenv("VISION_BACKEND", "blip").strip().lower()

    BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
    BLIP_MAX_CAPTION_LEN = int(os.getenv("BLIP_MAX_CAPTION_LEN", "140"))
    BLIP_MIN_CAPTION_LEN = int(os.getenv("BLIP_MIN_CAPTION_LEN", "18"))
    BLIP_NUM_BEAMS = int(os.getenv("BLIP_NUM_BEAMS", "5"))
    BLIP_REPETITION_PENALTY = float(os.getenv("BLIP_REPETITION_PENALTY", "1.22"))
    BLIP_LENGTH_PENALTY = float(os.getenv("BLIP_LENGTH_PENALTY", "1.12"))
    BLIP_USE_DUAL_CAPTION = os.getenv("BLIP_USE_DUAL_CAPTION", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    BLIP_CONDITIONAL_PREFIX = os.getenv(
        "BLIP_CONDITIONAL_PREFIX",
        "a detailed photograph showing",
    )

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    OLLAMA_TIMEOUT = 120
    OLLAMA_DESCRIPTION_MAX_TOKENS = int(os.getenv("OLLAMA_DESCRIPTION_MAX_TOKENS", "420"))
    OLLAMA_CHAT_MAX_TOKENS = int(os.getenv("OLLAMA_CHAT_MAX_TOKENS", "600"))
    OLLAMA_CHAT_TEMPERATURE = float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.72"))

    CHAT_MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_MAX_HISTORY_MESSAGES", "40"))
    CHAT_SYSTEM_PROMPT = os.getenv(
        "CHAT_SYSTEM_PROMPT",
        "You are the user's close friend texting them—warm, frank, a bit casual, no corporate "
        "or essay tone. Use contractions, short messages when it fits, and talk like a real person.\n\n"
        "If they send a photo plus a message: read what they actually asked or said. Answer that "
        "directly first. Do NOT turn the reply into a formal scene description, photo analysis, "
        "or social-media advice (no “you might post this on Instagram,” no lecturing about "
        "backdrops or poses) unless they clearly asked about those things.\n\n"
        "If a rough auto-caption about the image is provided, treat it as private context only—"
        "use it to understand the pic, but don't quote it or narrate the whole scene back unless "
        "they asked “what do you see” or want a description.\n\n"
        "Stay honest: if you're guessing from a vague caption, say so in a natural way. "
        "Don't invent stuff about the photo. Text-only messages: just chat like normal.",
    )
