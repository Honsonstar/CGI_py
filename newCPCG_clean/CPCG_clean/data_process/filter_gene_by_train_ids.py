#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_gene_by_train_ids.py

Given:
  1) gene csv (or csv.zip): rows are patients, first column is patient id (default: case_id)
  2) split csv: has a train column (or user-specified column), listing train patient ids

Output:
  A filtered gene csv containing only training patients (header included).
"""

import argparse
import os
import sys
import pandas as pd


def infer_train_col(split_df: pd.DataFrame) -> str:
    """Infer train column name from common patterns. Fallback to second column if needed."""
    cols = list(split_df.columns)
    # exact match
    if "train" in cols:
        return "train"
    # case-insensitive contains
    for c in cols:
        if str(c).strip().lower() == "train":
            return c
    for c in cols:
        if "train" in str(c).strip().lower():
            return c
    # fallback: user earlier mentioned "second column"; if exists use it
    if split_df.shape[1] >= 2:
        return cols[1]
    # otherwise first column
    return cols[0]


def read_table(path: str) -> pd.DataFrame:
    """Read CSV or CSV-in-ZIP with pandas. compression='infer' handles .zip."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, compression="infer", low_memory=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_csv", required=True, help="Path to gene file, e.g. tcga_gbmlgg_all_clean.csv.zip")
    ap.add_argument("--split_csv", required=True, help="Path to split file, e.g. splits_0.csv")
    ap.add_argument("--out_csv", required=True, help="Output path, e.g. tcga_gbmlgg_all_clean_splits_0.csv.zip")
    ap.add_argument("--id_col", default="case_id", help="Patient ID column name in gene CSV (default: case_id)")
    ap.add_argument("--train_col", default=None, help="Train column name in split CSV (default: auto-infer)")
    args = ap.parse_args()

    # 1) Load split and extract train IDs
    split_df = read_table(args.split_csv)
    train_col = args.train_col or infer_train_col(split_df)

    if train_col not in split_df.columns:
        raise ValueError(
            f"Train column '{train_col}' not found in split CSV. Available columns: {list(split_df.columns)}"
        )

    train_ids = (
        split_df[train_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    train_ids = train_ids[train_ids != ""].unique().tolist()
    train_id_set = set(train_ids)

    if len(train_ids) == 0:
        raise ValueError("No train IDs extracted. Please check split CSV format / train column.")

    # 2) Load gene CSV
    gene_df = read_table(args.gene_csv)

    # 2.1) Ensure id_col exists; fallback to first column if not
    id_col = args.id_col
    if id_col not in gene_df.columns:
        fallback = gene_df.columns[0]
        print(
            f"[WARN] id_col '{id_col}' not found in gene CSV. Falling back to first column: '{fallback}'",
            file=sys.stderr
        )
        id_col = fallback

    # 3) Filter rows by train IDs
    gene_ids = gene_df[id_col].astype(str).str.strip()
    gene_df[id_col] = gene_ids

    filtered = gene_df[gene_df[id_col].isin(train_id_set)].copy()

    # 4) Sanity checks
    found_set = set(filtered[id_col].tolist())
    missing = sorted(list(train_id_set - found_set))
    print(f"[INFO] Split file: {args.split_csv}")
    print(f"[INFO] Train column used: {train_col}")
    print(f"[INFO] Train IDs extracted: {len(train_id_set)}")
    print(f"[INFO] Gene file: {args.gene_csv}")
    print(f"[INFO] Rows in gene file: {len(gene_df)}")
    print(f"[INFO] Rows after filtering: {len(filtered)}")

    if missing:
        # Only print the first few to avoid huge logs
        print(f"[WARN] Missing {len(missing)} train IDs not found in gene CSV. Example: {missing[:10]}", file=sys.stderr)

    # 5) Save
    # pandas will infer compression based on extension if compression="infer" in to_csv is not supported across versions;
    # so we explicitly set it for .zip outputs.
    out_lower = args.out_csv.lower()
    if out_lower.endswith(".zip"):
        filtered.to_csv(args.out_csv, index=False, compression="zip")
    else:
        filtered.to_csv(args.out_csv, index=False)

    print(f"[INFO] Saved filtered gene CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
