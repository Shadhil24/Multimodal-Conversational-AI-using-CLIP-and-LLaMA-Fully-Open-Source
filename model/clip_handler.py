import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import Config


# Comprehensive candidate labels for zero-shot image classification
CANDIDATE_LABELS = [
    # People
    "a person", "a man", "a woman", "a child", "two children", "two kids",
    "a baby", "a group of people", "a crowd", "a couple", "an elderly person",
    "a teenager",
    # Animals
    "a dog", "a cat", "a bird", "a horse", "a cow", "a sheep", "a pig",
    "a lion", "a tiger", "a bear", "a fox", "a rabbit", "a deer", "a fish",
    "a butterfly", "an elephant", "a monkey", "a penguin", "a duck",
    # Vehicles
    "a car", "a truck", "a bus", "a motorcycle", "a bicycle", "an airplane",
    "a boat", "a train", "a helicopter", "a ship", "a scooter",
    # Nature & Scenes
    "a mountain", "a beach", "a forest", "a river", "a lake", "an ocean",
    "a desert", "a field", "a garden", "a waterfall", "a sunset", "a sunrise",
    "snow", "rain", "clouds", "a clear sky", "a storm",
    # Urban & Architecture
    "a city", "a building", "a house", "a bridge", "a street", "a road",
    "a park", "a skyscraper", "a church", "a castle", "a tower",
    # Food & Drinks
    "food", "a meal", "a pizza", "a burger", "a salad", "fruit", "vegetables",
    "a cake", "coffee", "a drink", "sushi", "bread", "pasta",
    # Objects
    "a book", "a phone", "a laptop", "a chair", "a table", "a bed",
    "a flower", "a tree", "a plant", "a ball", "a bottle", "a cup",
    "a bag", "clothes", "shoes", "a watch", "glasses",
    # Activities & Sports
    "sports", "running", "swimming", "cycling", "playing football",
    "dancing", "cooking", "reading", "painting", "music",
    # Indoor / Outdoor
    "an indoor scene", "an outdoor scene", "a room", "a kitchen",
    "a bedroom", "a living room", "an office", "a classroom",
    # Art & Style
    "a painting", "a drawing", "abstract art", "a photograph",
    "black and white image", "a colorful image",
]


class CLIPHandler:
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
        print(f"[CLIP] Loading model '{Config.CLIP_MODEL_NAME}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        print("[CLIP] Model loaded successfully.")

    def analyze_image(self, image: Image.Image, top_k: int = None) -> list[dict]:
        """
        Run zero-shot classification on the image against CANDIDATE_LABELS.
        Returns top_k results sorted by confidence (descending).
        """
        top_k = top_k or Config.CLIP_TOP_K

        inputs = self.processor(
            text=CANDIDATE_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image          # shape: (1, num_labels)
            probs = logits.softmax(dim=1)[0]           # shape: (num_labels,)

        k = min(top_k, len(CANDIDATE_LABELS))
        top_probs, top_indices = probs.topk(k)

        results = [
            {
                "label": CANDIDATE_LABELS[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        return results
