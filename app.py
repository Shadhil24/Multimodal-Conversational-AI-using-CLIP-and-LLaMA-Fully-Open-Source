"""
Flask application: serves UI templates and registers the REST API blueprint.
All features are implemented via /api/* — the browser UI calls those endpoints only.
"""
from flask import Flask, render_template
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from blueprints.api import register_api
from config import Config
from extensions import preload_vision_models, vision_backend

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
register_api(app)


# ---------------------------------------------------------------------------
# UI (HTML only — logic lives in static JS calling /api/*)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/legacy")
def legacy():
    return render_template("legacy.html")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    from flask import jsonify
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    from flask import jsonify
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    from flask import jsonify
    return jsonify({"error": f"Internal server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vb = vision_backend()
    print("=" * 60)
    print("  ImageBot – Flask UI + /api")
    print(f"  Vision mode : {vb}")
    if vb in ("blip", "both"):
        print(f"  BLIP model  : {Config.BLIP_MODEL_NAME}")
    if vb in ("clip", "both"):
        print(f"  CLIP model  : {Config.CLIP_MODEL_NAME}")
    print(f"  LLM model   : {Config.OLLAMA_MODEL}")
    print(f"  Ollama URL  : {Config.OLLAMA_BASE_URL}")
    print("  API index   : GET http://localhost:5000/api/")
    print("  Chat UI     : http://localhost:5000/")
    print("=" * 60)
    preload_vision_models()
    app.run(debug=True, host="0.0.0.0", port=5000)
