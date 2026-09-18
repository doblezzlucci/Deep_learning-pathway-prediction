

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import mygene
import gseapy as gp


@dataclass
class SSGSEAConfig:
    library: str = "KEGG_2021_Human"         # e.g., "Hallmark_2020", "MSigDB_C2_2023"
    batch_size: int = 500
    min_size: int = 10
    max_size: int = 10000
    processes: int = 4
    score_type: Literal["ES", "NES"] = "NES" # choose which score to keep
    sample_norm: bool = False               # keeps sample IDs stable more often


def map_ensembl_to_symbol(
    df: pd.DataFrame,
    species: str = "human",
    drop_unmapped: bool = True
) -> pd.DataFrame:
    """
    df: samples x genes, columns are Ensembl IDs
    Returns df with columns renamed to gene symbols.
    """
    mg = mygene.MyGeneInfo()
    ensembl_ids = df.columns.astype(str).tolist()

    mapping = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species=species,
        as_dataframe=False
    )

    map_df = pd.DataFrame(mapping)
    map_df = map_df.dropna(subset=["symbol"])[["query", "symbol"]]
    map_df = map_df.drop_duplicates(subset="query")

    rename_dict = dict(zip(map_df["query"], map_df["symbol"]))
    df2 = df.rename(columns=rename_dict)

    if drop_unmapped:
        df2 = df2.loc[:, df2.columns.isin(rename_dict.values())]

    return df2


def compute_ssgsea_scores(
    df_x: pd.DataFrame,
    out_path: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
    cfg: SSGSEAConfig = SSGSEAConfig(),
) -> pd.DataFrame:
    """
    df_x: samples x genes (Ensembl IDs as columns, sample IDs as index)
    out_path: if provided, saves to .csv or .parquet depending on extension
    drop_cols: list of Ensembl IDs to drop before mapping
    cfg: SSGSEAConfig

    Returns: pathway_scores (samples x pathways)
    """

    df = df_x.copy()

    # Ensure index is sample IDs (TCGA)
    df.index = df.index.astype(str)

    # Optional drop (your BRCA genes, etc.)
    if drop_cols:
        drop_cols = [str(c) for c in drop_cols]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print("Raw matrix (samples x genes):", df.shape)

    # Map Ensembl -> symbols
    df = map_ensembl_to_symbol(df, species="human", drop_unmapped=True)
    print("Mapped matrix (samples x symbols):", df.shape)

    # GSEAPY wants genes x samples
    expr = df.T
    expr.index.name = "Gene"
    print("Matrix for GSEAPY (genes x samples):", expr.shape)

    # Load gene set library
    gmt = gp.get_library(cfg.library)
    print(f"Loaded gene sets: {cfg.library}")

    # Batch by sample columns (keeps TCGA IDs)
    samples = expr.columns.tolist()
    num_batches = math.ceil(len(samples) / cfg.batch_size)
    print(f"Total samples: {len(samples)} → {num_batches} batches")

    all_batches = []

    for i in range(num_batches):
        start = i * cfg.batch_size
        end = start + cfg.batch_size
        batch_samples = samples[start:end]

        print(f"▶ Batch {i+1}/{num_batches} ({len(batch_samples)} samples)")

        batch_expr = expr[batch_samples]  # genes x batch_samples (TCGA IDs preserved)

        ssgsea_res = gp.ssgsea(
            data=batch_expr,
            gene_sets=gmt,
            outdir=None,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            permutation_num=0,
            processes=cfg.processes,
            sample_norm=cfg.sample_norm
        )

        # GSEApy returns res2d in a wide-ish format; try to normalize it to:
        # pathways x samples
        res = ssgsea_res.res2d.copy()

        # If already pathways x samples, keep it
        # Otherwise, handle common "long" format with columns Term/ES/NES/Name
        if {"Term", "ES", "NES"}.issubset(res.columns):
            # Some versions include "Name" or sample column elsewhere;
            # Usually res contains repeated blocks; safest approach:
            # Use the index as pathways if present, else use Term.
            # Here we assume long-format with an explicit sample column named "Name".
            if "Name" not in res.columns:
                raise ValueError(
                    "GSEApy output does not contain a 'Name' column. "
                    "Please print(ssgsea_res.res2d.head()) and I’ll adapt the parser."
                )
            score_col = cfg.score_type
            batch_scores = res.pivot(index="Term", columns="Name", values=score_col)
        else:
            # Assume it's already pathways x samples
            batch_scores = res

        # Ensure columns are the batch sample IDs (TCGA)
        # If columns are numeric indices, remap using batch_samples
        if all(isinstance(c, (int, float, np.integer)) for c in batch_scores.columns):
            # fallback remap
            if len(batch_scores.columns) == len(batch_samples):
                batch_scores.columns = batch_samples

        all_batches.append(batch_scores)

    # Merge batches (pathways x samples)
    pathway_by_sample = pd.concat(all_batches, axis=1)

    # Make columns unique just in case
    pathway_by_sample = pathway_by_sample.loc[:, ~pathway_by_sample.columns.duplicated()]

    # Transpose to samples x pathways
    pathway_scores = pathway_by_sample.T
    print("Final pathway matrix (samples x pathways):", pathway_scores.shape)

    # Save if requested
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()

        if ext == ".csv":
            pathway_scores.to_csv(out_path)
        elif ext in [".parquet", ".pq"]:
            pathway_scores.to_parquet(out_path, engine="pyarrow")
        else:
            raise ValueError("out_path must end with .csv or .parquet/.pq")

        print("Saved:", out_path)

    return pathway_scores
