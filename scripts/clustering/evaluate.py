#!/usr/bin/env python3
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
RESOLUTIONS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def spatial_smoothing(labels, coordinates, n_neighbors=6, n_iter=2):
    from scipy.sparse import csr_matrix

    _, idx = KDTree(coordinates).query(coordinates, k=n_neighbors + 1)
    labels = np.asarray(labels).astype(str)
    categories = np.sort(np.unique(labels))
    codes = pd.Categorical(labels, categories=categories).codes
    rows = np.repeat(np.arange(len(labels)), n_neighbors + 1)
    adjacency = csr_matrix((np.ones(rows.size, dtype=np.float32),
                            (rows, idx.ravel())), shape=(len(labels), len(labels)))
    for _ in range(n_iter):
        onehot = csr_matrix((np.ones(len(labels), dtype=np.float32),
                             (np.arange(len(labels)), codes)),
                            shape=(len(labels), len(categories)))
        codes = (adjacency @ onehot).toarray().argmax(axis=1)
    return categories[codes]


def evaluate(embedding, ground_truth, coordinates):
    adata = sc.AnnData(np.zeros((embedding.shape[0], 1)))
    adata.obsm["X_emb"] = StandardScaler().fit_transform(embedding)
    sc.pp.neighbors(adata, n_neighbors=20, use_rep="X_emb")
    best_res, best_ari, best_labels = None, -1.0, None
    for resolution in RESOLUTIONS:
        key = f"leiden_{resolution}"
        sc.tl.leiden(adata, resolution=resolution, random_state=42, key_added=key)
        labels = adata.obs[key].astype(str).values
        ari = adjusted_rand_score(ground_truth, labels)
        if ari > best_ari:
            best_res, best_ari, best_labels = resolution, ari, labels
    smoothed = spatial_smoothing(best_labels, coordinates)
    return {
        "leiden_smooth_res": best_res,
        "leiden_smooth_ari": adjusted_rand_score(ground_truth, smoothed),
        "leiden_smooth_nmi": normalized_mutual_info_score(ground_truth, smoothed),
    }


def read_array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    frame = pd.read_csv(path)
    return frame.iloc[:, 0].values if frame.shape[1] == 1 else frame.values


def main():
    parser = argparse.ArgumentParser(description="Evaluate spatial embeddings with Leiden clustering and spatial smoothing.")
    parser.add_argument("--embeddings", required=True, help="NPY or CSV matrix, spots x features")
    parser.add_argument("--coordinates", required=True, help="NPY or CSV matrix, spots x 2")
    parser.add_argument("--labels", required=True, help="NPY or one-column CSV ground-truth labels")
    parser.add_argument("--output", required=True, help="Output CSV for the one-row result")
    parser.add_argument("--model", default="model")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--sample", default="sample")
    parser.add_argument("--embedding", default="frozen", choices=["frozen", "finetuned", "specialized"])
    args = parser.parse_args()

    embedding = np.asarray(read_array(args.embeddings), dtype=np.float32)
    coordinates = np.asarray(read_array(args.coordinates), dtype=float)
    labels = np.asarray(read_array(args.labels)).astype(str).ravel()
    if embedding.ndim != 2 or coordinates.shape != (len(labels), 2) or len(embedding) != len(labels):
        raise ValueError("Expected embeddings (n,d), coordinates (n,2), and labels (n,), with identical n")
    valid = pd.notna(labels) & (labels != "nan")
    if not valid.all():
        embedding, coordinates, labels = embedding[valid], coordinates[valid], labels[valid]
    result = evaluate(embedding, labels, coordinates)
    row = {"model": args.model, "dataset": args.dataset, "sample": args.sample,
           "embedding": args.embedding, **result}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output, index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
