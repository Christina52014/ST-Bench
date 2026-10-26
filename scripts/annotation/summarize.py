#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rebuild the unified annotation summary, including archived kNN and LR rows.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    folder = args.root / "tables" / "annotation"
    reference = {"Xenium_Breast": "scFFPE_scRNA", "Xenium_breast": "scFFPE_scRNA",
                 "MERFISH": "BrainAgingSpatialAtlas_snRNA", "MERFISH_mouse_brain": "BrainAgingSpatialAtlas_snRNA"}
    canonical = {"Xenium_Breast": "Xenium_Breast", "Xenium_breast": "Xenium_Breast",
                 "MERFISH": "MERFISH_mouse_brain", "MERFISH_mouse_brain": "MERFISH_mouse_brain"}
    records = []
    with (folder / "annotation_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            dataset = canonical[row["dataset"]]
            records.append({"model": row["model"], "dataset": dataset, "method": "kNN",
                            "accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"]),
                            "reference": reference[dataset], "n_ref": "", "n_target": "",
                            "note": row.get("note", "")})
    sources = [("HEIST", folder / "sources/heist_annotation_summary.csv"),
               ("AIDO.Tissue", folder / "sources/aido_annotation_summary.csv")]
    for model, path in sources:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                dataset = canonical[row["dataset"]]
                records.append({"model": model, "dataset": dataset, "method": row["method"],
                                "accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"]),
                                "reference": reference[dataset],
                                "n_ref": int(row["n_ref"]) if row["n_ref"] else "",
                                "n_target": int(row["n_target"]) if row["n_target"] else "", "note": ""})
    model_order = ["HEIST", "AIDO.Tissue", "scGPT-spatial", "Novae", "OmniCell", "Nicheformer", "SToFM"]
    dataset_order = ["Xenium_Breast", "MERFISH_mouse_brain"]
    records.sort(key=lambda row: (dataset_order.index(row["dataset"]), model_order.index(row["model"])))
    columns = ["model", "dataset", "method", "accuracy", "macro_f1", "reference", "n_ref", "n_target", "note"]
    with (folder / "annotation_unified_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "accuracy": f'{record["accuracy"]:.6f}',
                             "macro_f1": f'{record["macro_f1"]:.6f}'})
    print(f'Wrote {len(records)} rows to {folder / "annotation_unified_summary.csv"}')


if __name__ == "__main__":
    main()
