#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Rebuild the unified gene-expression summary from released tables.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    folder = args.root / "tables" / "gene_expression"
    values = defaultdict(list)
    metrics = pd.read_csv(folder / "gene_expression_all_metrics.csv")
    for row in metrics[metrics["model"].isin(["STPath", "OmiCLIP"])].itertuples():
        values[(row.model, row.dataset)].append(float(row.global_r_all))
    specialized = pd.read_csv(folder / "gene_expression_summary_4x6.csv")
    for row in specialized.itertuples():
        for model in ["HisToGene", "BLEEP"]:
            value = getattr(row, model)
            if pd.notna(value) and str(value).upper() != "N/A":
                values[(model, row.dataset)].append(float(value))
    methods = ["STPath", "OmiCLIP", "HisToGene", "BLEEP"]
    datasets = ["DLPFC", "breast_cancer", "human_heart", "human_lymph_node", "mouse_brain", "mouse_kidney"]
    rows = []
    for dataset in datasets:
        row = {"dataset": dataset}
        for model in methods:
            current = values.get((model, dataset), [])
            row[model] = sum(current) / len(current) if current else float("nan")
        rows.append(row)
    result = pd.DataFrame(rows, columns=["dataset"] + methods)
    result.to_csv(folder / "gene_expression_unified_summary.csv", index=False,
                  float_format="%.6f", lineterminator="\r\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
