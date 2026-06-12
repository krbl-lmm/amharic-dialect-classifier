import joblib
import pandas as pd

from src.feature_extraction import extract_features_from_file

MODEL_PATH = "models/dialect_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_audio(audio_path):

    features = extract_features_from_file(audio_path)

    features = features.reshape(1, -1)

    probabilities = model.predict_proba(features)[0]

    classes = model.classes_

    results = {}

    for dialect, prob in zip(
        classes,
        probabilities
    ):
        results[dialect] = float(prob)

    predicted = max(
        results,
        key=results.get
    )

    return predicted, results