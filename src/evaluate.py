import argparse
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="metrics/eval.json")
    args = parser.parse_args()

    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    clf = joblib.load(args.model)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1}, f, indent=2)
    print("Metrics saved to", args.out)
