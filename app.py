# ========================================
# TruthGuard Flask API — mDeberta Version
# ========================================

import os
import time
import traceback
import joblib
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
# NEW: Hugging Face imports for mDeberta embedding
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)

print(">>> Starting TruthGuard app")

# ----------------- Paths -----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_LR = os.path.join(BASE_DIR, "model_lr_balanced.pkl")
MODEL_RF = os.path.join(BASE_DIR, "model_rf_balanced.pkl")
MODEL_ENSEMBLE = os.path.join(BASE_DIR, "model_ensemble.pkl")
EMBEDDER_MODEL_PATH = os.path.join(BASE_DIR, "embedder_model_name.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports.csv")

# ----------------- Globals -----------------
model_lr = None
model_rf = None
model = None # 'model' will be ensemble_model
tokenizer = None
embedder = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128 # Must match training script
print(f"Inference device: {DEVICE}")


# ----------------- Utility Functions (mDeberta Embedding) -----------------
def get_mdeberta_embedding(text):
    """Generates the mDeberta [CLS] embedding for a single text."""
    if tokenizer is None or embedder is None:
        raise RuntimeError("Embedder/Tokenizer not loaded.")

    embedder.eval()
    inputs = tokenizer(
        [text],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )
    # Move inputs to the correct device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embedder(**inputs)
    # Use CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding # Returns a numpy array of shape (1, 768)

# ----------------- Load Artifacts -----------------
def load_artifacts():
    """Load model(s), tokenizer, and embedder safely."""
    global model_lr, model_rf, model, tokenizer, embedder
    try:
        # Load mDeberta components
        if os.path.exists(EMBEDDER_MODEL_PATH):
            model_name = joblib.load(EMBEDDER_MODEL_PATH)
            print(f">>> Loading embedder: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            embedder = AutoModel.from_pretrained(model_name).to(DEVICE)
            print(f"✅ Embedder/Tokenizer loaded on {DEVICE}.")
        else:
            print("⚠️ Embedder model name not found. Cannot perform inference.")

        # Load Classification Models
        if os.path.exists(MODEL_LR):
            model_lr = joblib.load(MODEL_LR)
            print("✅ Logistic Regression model loaded.")
        if os.path.exists(MODEL_RF):
            model_rf = joblib.load(MODEL_RF)
            print("✅ Random Forest model loaded.")
        if os.path.exists(MODEL_ENSEMBLE): # Load the ensemble model into 'model' global
            model = joblib.load(MODEL_ENSEMBLE)
            print("✅ Ensemble model loaded.")

    except Exception as e:
        print("❌ Error loading artifacts:", e)
        traceback.print_exc()

load_artifacts()


# ----------------- Translator (Same) -----------------
try:
    from googletrans import Translator
    translator = Translator()
    print("🌐 Translator ready.")
except Exception:
    translator = None
    print("⚠️ googletrans not installed — translation disabled.")

# ----------------- Utility Functions (Same) -----------------
def map_confidence_to_label(prob):
    """Readable label mapping."""
    p = float(prob)
    if p <= 0.40:
        return "Likely Not Misleading"
    elif p <= 0.60:
        return "Uncertain / Needs Verification"
    else:
        return "Likely Misleading"

# ----------------- Routes -----------------
@app.route('/')
def home():
    # Assuming '1_api.html' is in the 'static' folder
    return app.send_static_file('1_api.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Predict misinformation likelihood using mDeberta embeddings."""
    global model_lr, model_rf, model, tokenizer, embedder

    data = request.get_json(force=True)
    text = data.get("text", "")
    model_choice = data.get("model", "rf")
    translate_flag = data.get("translate", False)

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text must be a non-empty string"}), 400

    if embedder is None or tokenizer is None:
        return jsonify({"error": "Embedder/Tokenizer missing. Please check model load."}), 500

    # Translate if requested
    text_for_pred = text
    if translate_flag and translator:
        try:
            translated = translator.translate(text, dest='en')
            text_for_pred = translated.text
        except Exception:
            print("⚠️ Translation failed, using original text.")

    try:
        # Step 1: Generate mDeberta Embedding
        X = get_mdeberta_embedding(text_for_pred)
        res = {}

        # --- choose model ---
        if model_choice == "lr" and model_lr is not None:
            prob = float(model_lr.predict_proba(X)[0][1])
            res["model_used"] = "LogisticRegression"
        # Use 'ensemble' as the model choice for the VotingClassifier
        elif model_choice == "ensemble" and model is not None:
            prob = float(model.predict_proba(X)[0][1])
            res["model_used"] = "Ensemble"
        elif model_rf is not None: # Default to Random Forest
            prob = float(model_rf.predict_proba(X)[0][1])
            res["model_used"] = "RandomForest"
        else:
            return jsonify({"error": "Requested model not available."}), 400

        res["confidence"] = prob
        res["label"] = map_confidence_to_label(prob)

        # Include both model confidences if both exist
        if model_lr and model_rf:
            res["confidence_lr"] = float(model_lr.predict_proba(X)[0][1])
            res["confidence_rf"] = float(model_rf.predict_proba(X)[0][1])

        return jsonify(res)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


@app.route("/report", methods=["POST"])
def report():
    """Record feedback into CSV."""
    data = request.get_json(force=True)
    text = str(data.get("text", "")).replace("\n", " ").replace("\r", " ")
    model_used = data.get("model_used", "")
    confidence = data.get("confidence", "")
    user_label = data.get("user_label", "")
    notes = data.get("notes", "")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    row = [ts, model_used, confidence, user_label, notes, text]

    try:
        header_needed = not os.path.exists(REPORT_PATH)
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,model_used,confidence,user_label,notes,text\n")

            def esc(s): return '"' + str(s).replace('"', '""') + '"'
            f.write(",".join([esc(x) for x in row]) + "\n")

        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Could not save report", "detail": str(e)}), 500


# ----------------- Main -----------------
if __name__ == "__main__":
    print("🚀 Starting TruthGuard API on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)