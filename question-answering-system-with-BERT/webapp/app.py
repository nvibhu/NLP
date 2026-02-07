import os
import sys
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is on sys.path so we can import src.* when running this app directly
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.infer import QAInference  # type: ignore


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")

    model_dir = os.environ.get("QA_MODEL_DIR", os.path.join(PROJECT_ROOT, "models", "qa-bert"))
    if not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        # Fallback to base model if fine-tuned model not found
        model_dir = None

    qa = QAInference(model_dir=model_dir)

    @app.route("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        context = data.get("context", "")
        question = data.get("question", "")
        if not context or not question:
            return jsonify({"error": "Both 'context' and 'question' are required."}), 400
        out = qa.predict(context, question)
        return jsonify({"answer": out["answer"], "score": out["score"]})

    @app.route("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(app.static_folder, filename)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
