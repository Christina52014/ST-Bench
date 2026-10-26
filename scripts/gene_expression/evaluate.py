#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def per_gene_correlation(predicted, measured):
    predicted = np.asarray(predicted, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    pc = predicted - predicted.mean(axis=0)
    gc = measured - measured.mean(axis=0)
    numerator = (pc * gc).sum(axis=0)
    denominator = np.sqrt((pc ** 2).sum(axis=0) * (gc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = numerator / denominator
    correlation[denominator == 0] = np.nan
    return correlation


def main():
    parser = argparse.ArgumentParser(description="Evaluate flattened global and per-gene Pearson correlations.")
    parser.add_argument("--predicted", required=True, help="NPY matrix, spots x genes")
    parser.add_argument("--measured", required=True, help="NPY matrix with the same shape and alignment")
    parser.add_argument("--output", required=True, help="JSON summary")
    parser.add_argument("--per-gene-output", help="Optional per-gene CSV")
    parser.add_argument("--gene-names", help="Optional one-column CSV without a header")
    args = parser.parse_args()

    predicted = np.load(args.predicted)
    measured = np.load(args.measured)
    if predicted.ndim != 2 or predicted.shape != measured.shape:
        raise ValueError("Predicted and measured matrices must be 2-D and have identical shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(measured).all():
        raise ValueError("Input matrices contain NaN or infinite values")
    global_r = float(pearsonr(predicted.ravel(), measured.ravel())[0])
    per_gene = per_gene_correlation(predicted, measured)
    result = {"n_spots": predicted.shape[0], "n_genes": predicted.shape[1],
              "global_r": global_r, "mean_per_gene_r": float(np.nanmean(per_gene))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.per_gene_output:
        if args.gene_names:
            names = pd.read_csv(args.gene_names, header=None).iloc[:, 0].astype(str).values
            if len(names) != predicted.shape[1]:
                raise ValueError("Gene-name count differs from the matrix")
        else:
            names = np.arange(predicted.shape[1]).astype(str)
        pd.DataFrame({"gene": names, "pearson_r": per_gene}).to_csv(args.per_gene_output, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
