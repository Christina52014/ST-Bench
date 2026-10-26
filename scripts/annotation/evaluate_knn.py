#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def read_labels(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).astype(str).ravel()
    return pd.read_csv(path).iloc[:, 0].astype(str).values


def main():
    parser = argparse.ArgumentParser(description="Reference-based cell-type annotation with distance-weighted kNN.")
    parser.add_argument("--reference-embeddings", required=True)
    parser.add_argument("--reference-labels", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-labels", help="Optional one-column CSV/NPY ground truth")
    parser.add_argument("--output", required=True, help="Prediction CSV")
    parser.add_argument("--metrics", help="Optional JSON metrics output")
    parser.add_argument("-k", type=int, default=15)
    args = parser.parse_args()

    reference = np.load(args.reference_embeddings)
    target = np.load(args.target_embeddings)
    reference_labels = read_labels(args.reference_labels)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise ValueError("Reference and target embeddings must be 2-D with the same feature dimension")
    if len(reference) != len(reference_labels):
        raise ValueError("Reference embedding and label counts differ")
    classifier = KNeighborsClassifier(n_neighbors=args.k, weights="distance")
    classifier.fit(reference, reference_labels)
    predictions = classifier.predict(target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"cell_id": np.arange(len(target)).astype(str), "pred": predictions})
    metrics = {"n_reference": len(reference), "n_target": len(target), "k": args.k}
    if args.target_labels:
        truth = read_labels(args.target_labels)
        if len(truth) != len(target):
            raise ValueError("Target embedding and label counts differ")
        result["true"] = truth
        metrics.update(accuracy=accuracy_score(truth, predictions),
                       macro_f1=f1_score(truth, predictions, average="macro", zero_division=0))
    result.to_csv(output, index=False)
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
