# ST-Bench

**A comprehensive benchmark of foundation models across four spatial transcriptomics tasks**

[Paper (PDF)](./paper/st-bench.pdf) · [Result tables](./tables/)

This repository contains all result tables and per-sample metrics accompanying the preprint:

> Mao Y., Sha C. *A comprehensive benchmark of foundation models across four spatial transcriptomics tasks* (2026).

We benchmark **9 spatial-omics foundation models** — scGPT-spatial, Novae, OmniCell, Nicheformer, OmiCLIP, SToFM, STPath, AIDO.Tissue-60M, HEIST — in frozen and fine-tuned modes against **7 specialized baselines** — GraphST, STAGATE, SpaGCN, HisToGene, BLEEP, Tangram, cell2location — on **4 tasks**, across **16 datasets, 6 platforms, and >700,000 cells/spots**, under unified downstream protocols.

## Key results at a glance

**No model dominates.** Each task has its own winner:

| Task | Dataset | Overall winner | Best FM | Key numbers |
|---|---|---|---|---|
| Clustering | DLPFC | GraphST | STPath (ft) | ARI 0.488 vs 0.349 |
| Clustering | Breast cancer | STAGATE | Novae (ft) | ARI 0.651 vs 0.603 |
| Clustering | Mouse brain | STAGATE | Novae (ft) | ARI 0.425 vs 0.390 |
| Clustering | MERFISH whole brain | (FM wins) | HEIST (ft) ≈ Nicheformer (ft) | ARI 0.241 |
| Clustering | BaristaSeq | (FM wins) | Novae (frozen) | ARI 0.655 |
| Annotation | Xenium breast | — | OmniCell | Acc/F1 0.763 / 0.541 |
| Annotation | MERFISH mouse brain | — | Nicheformer | Acc/F1 0.493 / 0.313 |
| Gene expression | all six datasets | — | OmiCLIP | global r 0.59–0.81, 6/6 wins |
| Deconvolution | DLPFC (5-layer) | GraphST | Novae | PCC 0.442 vs 0.294 |
| Deconvolution | Breast cancer | cell2location | Novae | PCC 0.423 vs 0.203 |
| Deconvolution | Lymph node | cell2location | SToFM | PCC 0.483 vs 0.302 |
| Deconvolution | Mouse brain anterior | Tangram | Novae | PCC 0.234 vs 0.098 |
| Deconvolution | Simulated seqFISH+ | Tangram | Novae | PCC 0.620 vs 0.388 |

## Main findings

- **Modality priors — not model scale — determine task fit.** Pathology-guided models (OmiCLIP, STPath) dominate wherever H&E images are available but collapse without them; scRNA+ST models (OmniCell, Nicheformer) win reference-based annotation; the graph-structured Novae wins no single task yet is the most consistent FM overall. The two newest generalist FMs (AIDO.Tissue-60M, HEIST) rank last.
- **Specialized methods are not yet displaced.** STAGATE, GraphST, and SpaGCN take the top three clustering positions; GraphST's deconvolution margin over the best FM (PCC 0.442 vs 0.294) exceeds the spread among all FMs.
- **Fine-tuning helps on average but can reverse sharply.** Mean ARI improves on all five clustering datasets (+0.02 to +0.09), yet the same model can swing in opposite directions (Nicheformer: +0.187 on MERFISH, −0.118 on BaristaSeq). Whether to fine-tune is a data-dependent decision.
- **Protocols shape conclusions.** We correct three circulating claims that invert under unified evaluation: a single-donor self-evaluation masquerading as cross-donor transfer, a label-mapping artifact creating a phantom deconvolution champion, and global correlation metrics confounded by evaluation gene-set size (up to 72% inflation).

If a single FM must be chosen for all tasks, **Novae offers the best worst-case performance**.

## Repository structure

```
paper/
  st-bench.pdf            Latest preprint
scripts/
  clustering/             Leiden + spatial-smoothing evaluator and summaries
  annotation/             Distance-weighted kNN evaluator and summaries
  gene_expression/        Global/per-gene correlation evaluator and summaries
  deconvolution/          Cosine-softmax and corrected DLPFC five-layer evaluators
  figures/                Final main-figure scripts
figures/                  The 12 PDF figures cited in the paper's main text
tables/
  clustering/             ARI/NMI per (model, mode, dataset), with specialized baselines
  annotation/             Accuracy / Macro-F1 summaries + per-cell predictions (Xenium, MERFISH)
  gene_expression/        Global r / per-gene r summaries, panel-sensitivity analyses
  deconv/                 DLPFC 5-layer & extension-dataset tables, per-sample cosine/NNLS results
  dlpfc_per_sample/       Per-sample deconvolution and gene-expression summaries
```

## Reproducibility

Summary tables and selected main-figure panels can be rebuilt directly from the released CSVs. The task evaluators also accept new embeddings and predictions supplied by users. Raw datasets, model checkpoints, and model-specific training environments are external.

See the [analysis and reproduction guide](./scripts/README.md) for installation, commands, input formats, and evaluation protocols.

## Data availability

Raw datasets are all public; see the preprint Methods (Table S1) for sources and accession links. Fine-tuning hyperparameters for every foundation model are listed in Table S2.

## Citation

If you use these results in your work, please cite:

```bibtex
@article{mao2026stbench,
  title  = {A comprehensive benchmark of foundation models
            across four spatial transcriptomics tasks},
  author = {Mao, Yicong and Sha, Chulin},
  year   = {2026},
  note   = {Preprint}
}
```

## License

(to be selected before public release)
