import os, sys, json, io, base64
from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_all_features, annotate_image
from disease_classifier import predict, DISEASES, get_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


# ── Pre-load model at startup ──────────────────────────────────────────────────
print("Loading ML model …", flush=True)
get_model()
print("Model ready.", flush=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, or WEBP."}), 400

    image_bytes = file.read()

    try:
        # Feature extraction
        features = extract_all_features(image_bytes)

        # Prediction
        class_id, confidence, all_probs = predict(features)
        disease = DISEASES[class_id]

        # Annotated image
        annotated_bytes = annotate_image(image_bytes)
        annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

        # Sort probabilities
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "success": True,
            "disease": {
                "name": disease["name"],
                "description": disease["description"],
                "severity": disease["severity"],
                "color": disease["color"],
                "icon": disease["icon"],
                "treatments": disease["treatments"],
                "prevention": disease["prevention"],
            },
            "confidence": round(confidence * 100, 1),
            "all_probabilities": sorted_probs[:5],
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/diseases")
def list_diseases():
    return jsonify([
        {"id": k, "name": v["name"], "severity": v["severity"], "color": v["color"]}
        for k, v in DISEASES.items()
    ])


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForest+Calibration"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)