import argparse
import pandas as pd
from sklearn.datasets import load_iris
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("target")], axis=1)
    df.to_csv(os.path.join(args.out_dir, "iris.csv"), index=False)
    print("Saved iris.csv in", args.out_dir)
