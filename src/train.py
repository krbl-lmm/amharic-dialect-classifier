from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.load_data import load_all_datasets
from src.feature_extraction import extract_features


def build_dataset(dev_mode=True):
    df = load_all_datasets(dev_mode=dev_mode)

    X = []
    y = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting features"
    ):
        features = extract_features(row["audio"])

        X.append(features)
        y.append(row["dialect"])

    return np.array(X), np.array(y)


def main():

    print("Loading dataset...")

    X, y = build_dataset(dev_mode=True)

    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=101,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=101,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "models/dialect_model.pkl"
    )

    print("\nModel saved.")


if __name__ == "__main__":
    main()