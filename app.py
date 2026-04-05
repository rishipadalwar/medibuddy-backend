"""
Medicine Recommendation System - REST API Backend
=================================================
Clean rewrite — reads symptom names directly from the model.
Handles numeric disease labels + fuzzy CSV name matching.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
from textblob import TextBlob
chat_history = []
fallback_memory = []



app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# 1. LOAD MODEL & DATASETS
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model         = pickle.load(open(os.path.join(BASE_DIR, "model", "RandomForest.pkl"), "rb"))
SYMPTOMS_LIST = model.feature_names_in_.tolist()

sym_des     = pd.read_csv(os.path.join(BASE_DIR, "dataset", "symptoms_df.csv"))
precautions = pd.read_csv(os.path.join(BASE_DIR, "dataset", "precautions_df.csv"))
workout     = pd.read_csv(os.path.join(BASE_DIR, "dataset", "workout_df.csv"))
description = pd.read_csv(os.path.join(BASE_DIR, "dataset", "description.csv"))
medications = pd.read_csv(os.path.join(BASE_DIR, "dataset", "medications.csv"))
diets       = pd.read_csv(os.path.join(BASE_DIR, "dataset", "diets.csv"))

# Build disease label → name map from Training.csv
training_data = pd.read_csv(os.path.join(BASE_DIR, "dataset", "Training.csv"))
DISEASE_NAMES = sorted(training_data['prognosis'].unique().tolist())

# ─────────────────────────────────────────
# NORMALIZE DISEASE NAMES IN ALL DATASETS
#
# Problem: the same disease can appear as:
#   "(vertigo) Paroymsal  Positional Vertigo"  ← double space, typo
#   "(vertigo) Paroxysmal Positional Vertigo"  ← different CSV
#
# Fix: strip + collapse multiple spaces + lowercase
# for matching purposes only (we keep original for display).
# ─────────────────────────────────────────
def normalize(name: str) -> str:
    """Lowercase, strip, collapse multiple spaces."""
    return " ".join(str(name).lower().split())


# Pre-build normalized lookup keys for each dataset
desc_map  = {normalize(r['Disease']): i for i, r in description.iterrows()}
prec_map  = {normalize(r['Disease']): i for i, r in precautions.iterrows()}
med_map   = {normalize(r['Disease']): i for i, r in medications.iterrows()}
diet_map  = {normalize(r['Disease']): i for i, r in diets.iterrows()}
work_map  = {normalize(r['disease']): i for i, r in workout.iterrows()}   # note lowercase 'disease'

print(f"[INFO] Model loaded. Symptoms: {len(SYMPTOMS_LIST)}, Diseases: {len(DISEASE_NAMES)}")


# ─────────────────────────────────────────
# 2. HELPER: symptoms → DataFrame
# ─────────────────────────────────────────

def symptoms_to_vector(user_symptoms: list) -> pd.DataFrame:
    vector    = {col: [0] for col in SYMPTOMS_LIST}
    matched   = []
    unmatched = []

    for symptom in user_symptoms:
        cleaned = symptom.strip().lower().replace(" ", "_")
        if cleaned in vector:
            vector[cleaned] = [1]
            matched.append(cleaned)
        else:
            unmatched.append(symptom)

    if unmatched:
        print(f"[WARN] Unrecognized symptoms: {unmatched}")
    print(f"[INFO] Matched: {matched}")

    df = pd.DataFrame(vector)
    print(f"[INFO] Input shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# 3. HELPER: recommendations lookup
# ─────────────────────────────────────────

def get_recommendations(disease: str) -> dict:
    key = normalize(disease)  # normalize for lookup

    # ── Description ──
    if key in desc_map:
        desc = str(description.loc[desc_map[key], 'Description'])
    else:
        desc = "No description available."
        print(f"[WARN] No description found for: '{disease}' (key: '{key}')")

    # ── Precautions ──
    if key in prec_map:
        row = precautions.loc[prec_map[key]]
        raw = [row['Precaution_1'], row['Precaution_2'],
               row['Precaution_3'], row['Precaution_4']]
        prec_list = [str(p) for p in raw if pd.notna(p) and str(p).strip()]
    else:
        prec_list = []
        print(f"[WARN] No precautions found for: '{disease}'")

    # ── Medications ──
    if key in med_map:
        # medications can have multiple rows per disease
        med_rows  = medications[medications['Disease'].apply(normalize) == key]
        med_list  = [str(m) for m in med_rows['Medication'].tolist()]
    else:
        med_list = []
        print(f"[WARN] No medications found for: '{disease}'")

    # ── Diet ──
    if key in diet_map:
        diet_rows = diets[diets['Disease'].apply(normalize) == key]
        diet_list = [str(d) for d in diet_rows['Diet'].tolist()]
    else:
        diet_list = []
        print(f"[WARN] No diet found for: '{disease}'")

    # ── Workout ──
    if key in work_map:
        work_rows = workout[workout['disease'].apply(normalize) == key]
        work_list = [str(w) for w in work_rows['workout'].tolist()]
    else:
        work_list = []
        print(f"[WARN] No workout found for: '{disease}'")

    return {
        "description": desc,
        "precautions": prec_list,
        "medications": med_list,
        "diet":        diet_list,
        "workout":     work_list,
    }


# ─────────────────────────────────────────
# 4. API ROUTES
# ─────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Medicine Recommendation API is running",
        "total_symptoms": len(SYMPTOMS_LIST),
        "total_diseases": len(DISEASE_NAMES)
    }), 200


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({
        "total":    len(SYMPTOMS_LIST),
        "symptoms": SYMPTOMS_LIST
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "symptoms": ["headache", "nausea", ...] }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    if "symptoms" not in data:
        return jsonify({"error": "Missing key 'symptoms'."}), 400

    user_symptoms = data["symptoms"]
    if not isinstance(user_symptoms, list) or len(user_symptoms) == 0:
        return jsonify({"error": "'symptoms' must be a non-empty list."}), 400

    input_vector      = symptoms_to_vector(user_symptoms)
    raw_prediction    = model.predict(input_vector)[0]
    label             = int(raw_prediction)
    predicted_disease = DISEASE_NAMES[label]

    print(f"[INFO] Label: {label} → Disease: '{predicted_disease}'")

    recommendations = get_recommendations(predicted_disease)

    return jsonify({
        "disease": predicted_disease,
        **recommendations
    }), 200


@app.route("/psychologist", methods=["POST"])
def psychologist():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text'"}), 400

    user_input = data["text"]
    
    # Store user message
    fallback_memory.append({"role": "user", "content": user_input})

    text = user_input.lower()
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    # Get previous message for context (last USER message only)
    previous = ""
    for msg in reversed(fallback_memory[:-1]):  
        if msg["role"] == "user":
            previous = msg["content"]
            break

    def build_response(emotion, intro, suggestions):
        reply_text = intro + "\n\nHere are a few things you could try:\n• " + "\n• ".join(suggestions)
        # Save bot reply
        fallback_memory.append({"role": "assistant", "content": reply_text})
        return {
            "emotion": emotion,
            "reply": reply_text,
            "source": "local_ai"
        }

    # 🔥 STRESS / ANXIETY
    if any(word in text for word in ["stress", "stressed", "anxious", "anxiety", "overwhelmed"]):
        return jsonify(build_response(
            "Stressed",
            f"It sounds like you're feeling overwhelmed. Earlier you mentioned: '{previous}'. I'm here with you — let's take this step by step.",
            ["Take 5 slow deep breaths (inhale 4s, exhale 6s)", "Break your work into smaller tasks", "Take a short break to reset your mind"]
        ))

    # 🔥 SAD / LOW
    elif any(word in text for word in ["sad", "depressed", "lonely", "tired", "exhausted", "down", "low"]):
        return jsonify(build_response(
            "Sad",
            f"I'm really sorry you're feeling this way. Earlier you said: '{previous}'. You don’t have to go through this alone — I’m here for you.",
            ["Talk to someone you trust", "Get proper rest — your body might need it", "Do something comforting like music or a walk"]
        ))

    # 🔥 ANGER
    elif any(word in text for word in ["angry", "frustrated", "irritated"]):
        return jsonify(build_response(
            "Angry",
            f"It sounds like something is really frustrating you. Earlier you mentioned: '{previous}'. Let’s try to slow things down together.",
            ["Take a pause before reacting", "Try deep breathing to calm your body", "Step away from the situation for a bit"]
        ))

    # 🔥 HAPPY
    elif any(word in text for word in ["happy", "excited", "good", "great"]):
        return jsonify(build_response(
            "Happy",
            f"That’s really nice to hear! 😊 Earlier you said: '{previous}'. It’s great that you're feeling this way.",
            ["Take a moment to enjoy this feeling", "Share your happiness with someone", "Keep doing what made you feel this way"]
        ))

    # 🔥 CONFUSED / LOST
    elif any(word in text for word in ["confused", "lost", "don't know", "uncertain"]):
        return jsonify(build_response(
            "Confused",
            f"It sounds like you're feeling unsure right now. Earlier you mentioned: '{previous}'. That can be really overwhelming.",
            ["Break things into smaller decisions", "Write down your thoughts to gain clarity", "Focus on what you can control right now"]
        ))

    # 🔥 FALLBACK TO SENTIMENT
    else:
        if polarity < -0.2:
            return jsonify(build_response(
                "Sad",
                f"I can sense something might be bothering you. Earlier you said: '{previous}'. I'm here to listen.",
                ["Take things one step at a time", "Express your thoughts freely", "Give yourself time to process things"]
            ))
        elif polarity > 0.3:
            return jsonify(build_response(
                "Positive",
                f"You seem to be in a positive mindset. Earlier you mentioned: '{previous}'. That’s great to see 😊",
                ["Keep nurturing this mindset", "Use this energy productively", "Spread positivity around you"]
            ))
        else:
            return jsonify(build_response(
                "Neutral",
                f"I’m here with you. Earlier you mentioned: '{previous}'. If you want to share more, I’m listening.",
                ["Take a moment to reflect", "Write down what you're feeling", "Take a short break to clear your mind"]
            ))

# ─────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
