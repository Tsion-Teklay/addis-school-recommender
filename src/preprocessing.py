import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from data_loader import load_raw_data, basic_validation


def ensure_processed_directory():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def preprocess():
    ensure_processed_directory()

    schools, parents, interactions = load_raw_data()
    basic_validation(schools, parents, interactions)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Merge parent-school pairs from interactions
    df = interactions.merge(parents, on="parent_id")
    df = df.merge(schools, on="school_id")

    # Features to use
    categorical_features = [
        "subcity_x",        # parent subcity
        "subcity_y",        # school subcity
        "preferred_stream",
        "stream",
        "preferred_type",
        "type"
    ]

    numerical_features = [
        "budget",
        "child_avg",
        "annual_fee",
        "ranking_score",
        "facilities_score",
        "teacher_quality"
    ]

    X = df[categorical_features + numerical_features]
    y = df["rating"]

    # -----------------------------
    # Preprocessing Pipeline
    # -----------------------------

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Save processed data
    np.save("data/processed/X.npy", X_processed)
    np.save("data/processed/y.npy", y.values)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Preprocessing complete. Processed data saved.")


if __name__ == "__main__":
    preprocess()