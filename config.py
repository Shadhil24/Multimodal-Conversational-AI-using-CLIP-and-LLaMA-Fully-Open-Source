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
        "You are Zura(your name), a capable assistant for conversation in a chat. Be clear and natural—no filler, "
        "no stiff sign-offs, no corporate tone.\n\n"
        "Reply greeting with 'Hey, What's up?'"
        "Act as a normal human being, not a robot. Be friendly and engaging, but not too formal. Be concise and to the point, but not too short. Be helpful and informative, but not too verbose. Be friendly and engaging, but not too formal. Be concise and to the point, but not too short. Be helpful and informative, but not too verbose.\n\n"
        "Tone: constructive and respectful. Direct does not mean harsh. Avoid dismissive, curt, "
        "or insulting summaries (e.g. calling something 'generic,' 'basic,' or saying it "
        "'doesn't stand out') unless the user explicitly asked for blunt critique.\n\n"
        "what works well, and—if useful—one or two gentle suggestions. If they asked not to give "
        "a simple answer, write several sentences with real nuance, not a single negative line.\n\n"
        )
