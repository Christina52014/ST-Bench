#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Rebuild clustering summary tables from the released full result table.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    folder = args.root / "tables" / "clustering"
    full = pd.read_csv(folder / "leiden_smooth_results_with_baselines.csv")
    full["mode"] = full["embedding"].where(full["embedding"].isin(["frozen", "finetuned"]), "specialized")
    for metric in ("ari", "nmi"):
        column = f"leiden_smooth_{metric}"
        summary = full.groupby(["dataset", "model", "mode"])[column].mean().unstack("mode").reset_index()
        summary = summary.sort_values(["dataset", "finetuned"], ascending=[True, False])
        summary.to_csv(folder / f"clustering_summary_{metric}_by_dataset_with_baselines.csv", index=False)
    per_dataset = full.groupby(["dataset", "model", "mode"])["leiden_smooth_ari"].mean().reset_index()
    per_dataset["rank"] = per_dataset.groupby("dataset")["leiden_smooth_ari"].rank(ascending=False)
    overall = per_dataset.groupby(["model", "mode"]).agg(mean_ari=("leiden_smooth_ari", "mean"),
        mean_rank=("rank", "mean"), n_datasets=("dataset", "nunique")).reset_index().sort_values("mean_rank")
    overall.to_csv(folder / "clustering_overall_ranking_with_baselines.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
