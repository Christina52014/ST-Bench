#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["Upper(L1-3)", "L4", "L5", "L6", "WM"]
GT_MERGE = {"Layer1": "Upper(L1-3)", "Layer2": "Upper(L1-3)", "Layer3": "Upper(L1-3)",
            "Layer4": "L4", "Layer5": "L5", "Layer6": "L6", "WM": "WM"}
TYPES33 = ["Astros_1", "Astros_2", "Astros_3", "Endo", "Ex_10_L2_4", "Ex_1_L5_6",
           "Ex_2_L5", "Ex_3_L4_5", "Ex_4_L_6", "Ex_5_L5", "Ex_6_L4_6", "Ex_7_L4_6",
           "Ex_8_L5_6", "Ex_9_L5_6", "Inhib_1", "Inhib_2_VIP", "Inhib_3_SST",
           "Inhib_4_SST", "Inhib_5", "Inhib_6_SST", "Inhib_7_PVALB", "Inhib_8_PVALB",
           "Micro/Macro", "Mix_1", "Mix_2", "Mix_3", "Mix_4", "Mix_5", "OPCs_1",
           "OPCs_2", "Oligos_1", "Oligos_2", "Oligos_3"]


def fine_weights(fine_type):
    weights = dict.fromkeys(CLASSES, 0.0)
    upper, l4, l5, l6 = "Upper(L1-3)", "L4", "L5", "L6"
    if fine_type.startswith("Ex_"):
        if "L2_4" in fine_type: weights[upper], weights[l4] = 2 / 3, 1 / 3
        elif "L4_5" in fine_type: weights[l4], weights[l5] = .5, .5
        elif "L4_6" in fine_type: weights[l4] = weights[l5] = weights[l6] = 1 / 3
        elif "L5_6" in fine_type: weights[l5], weights[l6] = .5, .5
        elif "L5" in fine_type: weights[l5] = 1.0
        elif "L_6" in fine_type or "L6" in fine_type: weights[l6] = 1.0
        elif "L4" in fine_type: weights[l4] = 1.0
        else: weights[upper] = 1.0
    elif fine_type.startswith("Inhib_"):
        if "VIP" in fine_type: weights[upper], weights[l4], weights[l5] = .7, .15, .15
        elif "SST" in fine_type: weights[upper], weights[l4], weights[l5], weights[l6] = .2, .2, .3, .3
        elif "PVALB" in fine_type: weights[l4], weights[l5], weights[l6] = .4, .4, .2
        else: weights[upper], weights[l4], weights[l5], weights[l6] = .35, .2, .25, .2
    elif fine_type.startswith("Oligos") or fine_type.startswith("OPCs"): weights["WM"] = 1.0
    elif fine_type.startswith("Astros"): weights["WM"], weights[upper], weights[l4], weights[l5], weights[l6] = .25, .25, .15, .2, .15
    elif fine_type.startswith("Micro") or fine_type == "Endo":
        for label in CLASSES: weights[label] = .2
    elif fine_type.startswith("Mix_"): weights[upper], weights[l4], weights[l5], weights[l6] = .4, .2, .2, .2
    else: weights[upper] = 1.0
    return weights


WEIGHTS = np.array([[fine_weights(label)[group] for group in CLASSES] for label in TYPES33])


def to_merged(probabilities, fine_types):
    indices = [TYPES33.index(label) for label in fine_types]
    merged = probabilities @ WEIGHTS[indices]
    return merged / (merged.sum(axis=1, keepdims=True) + 1e-10)


def compute_metrics(probabilities, ground_truth):
    ground_truth = pd.Series(ground_truth).astype(str).replace(GT_MERGE).values
    predicted_labels = np.array(CLASSES)[probabilities.argmax(axis=1)]
    onehot = np.zeros((len(ground_truth), len(CLASSES)))
    for i, label in enumerate(ground_truth):
        onehot[i, CLASSES.index(label)] = 1.0
    pcc = [np.corrcoef(probabilities[:, j], onehot[:, j])[0, 1]
           if np.std(probabilities[:, j]) > 1e-12 and 0 < onehot[:, j].sum() < len(ground_truth) else np.nan
           for j in range(len(CLASSES))]
    return {"pcc": float(np.nanmean(pcc)),
            "rmse": float(np.sqrt(np.mean((probabilities - onehot) ** 2))),
            "jsd": float(np.mean([jensenshannon(onehot[i] + 1e-10, probabilities[i] + 1e-10) ** 2
                                  for i in range(len(ground_truth))])),
            "accuracy": float(accuracy_score(ground_truth, predicted_labels)),
            "macro_f1": float(f1_score(ground_truth, predicted_labels, average="macro", zero_division=0))}


def main():
    parser = argparse.ArgumentParser(description="Evaluate 33-type DLPFC predictions in the corrected five-layer space.")
    parser.add_argument("--probabilities", required=True, help="CSV with 33 fine-type probability columns")
    parser.add_argument("--labels", required=True, help="One-column CSV/NPY with layer labels")
    parser.add_argument("--output", required=True, help="JSON metrics output")
    args = parser.parse_args()
    frame = pd.read_csv(args.probabilities)
    missing = set(frame.columns) - set(TYPES33)
    if missing:
        raise ValueError(f"Unexpected fine-type columns: {sorted(missing)}")
    merged = to_merged(frame.values, frame.columns)
    labels_path = Path(args.labels)
    labels = np.load(labels_path, allow_pickle=True) if labels_path.suffix == ".npy" else pd.read_csv(labels_path).iloc[:, 0].values
    if len(labels) != len(merged):
        raise ValueError("Probability and label counts differ")
    result = compute_metrics(merged, labels)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
