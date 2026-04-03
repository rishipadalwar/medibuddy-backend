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


# ─────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)