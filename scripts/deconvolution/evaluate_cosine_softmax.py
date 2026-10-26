#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score


def read_labels(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True).astype(str).ravel()
    return pd.read_csv(path).iloc[:, 0].astype(str).values


def predict(reference, reference_labels, target, temperature=5.0):
    cell_types = np.array(sorted(set(reference_labels)))
    centroids = np.stack([reference[reference_labels == label].mean(axis=0) for label in cell_types])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
    target = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-10)
    return cell_types, softmax((target @ centroids.T) * temperature, axis=1)


def evaluate(probabilities, cell_types, ground_truth):
    positions = {label: i for i, label in enumerate(cell_types)}
    missing = sorted(set(ground_truth) - set(positions))
    if missing:
        raise ValueError(f"Ground-truth classes missing from reference: {missing}")
    onehot = np.zeros_like(probabilities)
    for i, label in enumerate(ground_truth):
        onehot[i, positions[label]] = 1.0
    pcc = [np.corrcoef(probabilities[:, j], onehot[:, j])[0, 1]
           if np.std(probabilities[:, j]) > 1e-12 and 0 < onehot[:, j].sum() < len(ground_truth) else np.nan
           for j in range(len(cell_types))]
    predicted_labels = cell_types[probabilities.argmax(axis=1)]
    return {
        "pcc": float(np.nanmean(pcc)),
        "rmse": float(np.sqrt(np.mean((probabilities - onehot) ** 2))),
        "jsd": float(np.mean([jensenshannon(onehot[i] + 1e-10, probabilities[i] + 1e-10) ** 2
                              for i in range(len(ground_truth))])),
        "accuracy": float(accuracy_score(ground_truth, predicted_labels)),
        "macro_f1": float(f1_score(ground_truth, predicted_labels, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Cosine-softmax cell-type deconvolution from reference centroids.")
    parser.add_argument("--reference-embeddings", required=True)
    parser.add_argument("--reference-labels", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-labels", help="Optional one-column CSV/NPY ground truth")
    parser.add_argument("--probabilities", required=True, help="Output probability CSV")
    parser.add_argument("--output", help="Optional JSON metrics output")
    parser.add_argument("--temperature", type=float, default=5.0)
    args = parser.parse_args()

    reference = np.load(args.reference_embeddings)
    target = np.load(args.target_embeddings)
    reference_labels = read_labels(args.reference_labels)
    if reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        raise ValueError("Reference and target embeddings must be 2-D with the same feature dimension")
    if len(reference) != len(reference_labels):
        raise ValueError("Reference embedding and label counts differ")
    cell_types, probabilities = predict(reference, reference_labels, target, args.temperature)
    output = Path(args.probabilities)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probabilities, columns=cell_types).to_csv(output, index=False)
    metrics = {"n_reference": len(reference), "n_target": len(target),
               "temperature": args.temperature, "cell_types": cell_types.tolist()}
    if args.target_labels:
        ground_truth = read_labels(args.target_labels)
        if len(ground_truth) != len(target):
            raise ValueError("Target embedding and label counts differ")
        metrics.update(evaluate(probabilities, cell_types, ground_truth))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
