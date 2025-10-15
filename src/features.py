import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import re
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # Basic filtering and column selection for house price prediction
    if "purpose" in df.columns:
        df = df[df["purpose"].astype(str).str.lower() == "for sale"]

    # Ensure price is numeric and present
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])  # drop rows without price

    # Feature set: blend of numeric and categorical
    selected_features = [
        "property_type",
        "city",
        "province_name",
        "baths",
        "bedrooms",
        "Area Size",
        "Area Type",
        # you may add more: 'location', 'location_id', etc.
    ]

    available_features = [f for f in selected_features if f in df.columns]
    features_df = df[available_features].copy()

    # Prepare label encoders for object/categorical features
    label_encoders = {}
    for col in available_features:
        if features_df[col].dtype == object or str(features_df[col].dtype).startswith("category"):
            le = LabelEncoder()
            features_df[col] = features_df[col].astype(str)
            features_df[col] = le.fit_transform(features_df[col])
            label_encoders[col] = le
        else:
            # coerce numeric columns
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    # Drop rows with any missing feature after coercion
    features_df = features_df.dropna(axis=0)

    # Align target (price) with features_df index
    y = df.loc[features_df.index, "price"].values
    X = features_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    print("Train/test data saved in", args.out_dir)

    # Persist feature metadata required by the Flask app
    # Build ordered feature list from dataframe columns
    feature_list = list(features_df.columns)

    # Map feature name -> form field name expected by the app
    def sanitize_to_field(name: str) -> str:
        lowered = name.lower()
        # Replace non-alphanumeric with underscores, collapse repeats
        replaced = re.sub(r"[^0-9a-z]+", "_", lowered)
        replaced = re.sub(r"_+", "_", replaced).strip("_")
        return replaced

    feature_field_map = {feat: sanitize_to_field(feat) for feat in feature_list}

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(feature_list, os.path.join(models_dir, "model_features.pkl"))
    joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))
    joblib.dump(feature_field_map, os.path.join(models_dir, "feature_field_map.pkl"))
    print("Feature metadata saved in", models_dir)
