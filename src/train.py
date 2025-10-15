import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_out", type=str, default="models/house_price_model.pkl")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))

    reg = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    reg.fit(X_train, y_train)

    # Ensure output directory exists
    model_out_dir = os.path.dirname(os.path.abspath(args.model_out)) or "."
    os.makedirs(model_out_dir, exist_ok=True)

    joblib.dump(reg, args.model_out)
    print("Model saved to", args.model_out)
