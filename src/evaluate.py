import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="metrics/eval.json")
    args = parser.parse_args()

    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    reg = joblib.load(args.model)
    preds = reg.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"r2": r2, "mae": mae, "rmse": rmse}, f, indent=2)
    print("Metrics saved to", args.out)
