#!/usr/bin/env python3
"""
levels.py
────────────────────
Reads  ../data/COORDINATE_XY.csv  and  ../data/classification.csv, 
fits a Decision Tree (entropy, up to 5 leaves)
on the y_coord score to partition neurons into functional levels, and writes
../data/COORDINATE_XY_with_levels_tree.csv.

Columns in output:
  root_id   - neuron identifier
  x_coord   - Fiedler vector
  score     - y_coord renamed for clarity
  y_level   - functional level (1 = highest score, N = lowest)

Usage:
  python levels.py
  python levels.py --max-leaves 5 --min-samples 50
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def assign_levels(
    data_folder: str,
    max_leaves: int = 5,
    min_samples_leaf: int = 50,
    random_state: int = 42,
) -> None:
    start_time = time.time()

    coords_file = os.path.join(data_folder, "COORDINATE_XY.csv")
    class_file  = os.path.join(data_folder, "classification.csv")
    output_file = os.path.join(data_folder, "COORDINATE_XY_with_levels_tree.csv")

    # --- 1. Load data --------------------------------------------------------
    print(f"[INFO] Loading: {coords_file}")
    if not os.path.exists(coords_file):
        print(f"[ERROR] File not found: {coords_file}")
        print("[HINT] Run coordinates.py first.")
        sys.exit(1)

    print(f"[INFO] Loading: {class_file}")
    if not os.path.exists(class_file):
        print(f"[ERROR] File not found: {class_file}")
        sys.exit(1)

    coords         = pd.read_csv(coords_file)
    classification = pd.read_csv(class_file)

    for fname, df_check, cols in [
        (coords_file,  coords,         {"root_id", "x_coord", "y_coord"}),
        (class_file,   classification, {"root_id", "super_class"}),
    ]:
        missing = cols - set(df_check.columns)
        if missing:
            print(f"[ERROR] {fname} is missing columns: {missing}")
            sys.exit(1)

    print(f"[INFO] Coordinates loaded: {len(coords):,} neurons")
    print(f"[INFO] Classification loaded: {len(classification):,} entries")

    # --- 2. Merge and rename -------------------------------------------------
    df = coords.merge(
        classification[["root_id", "super_class"]],
        on="root_id",
        how="inner",
    )
    n_dropped = len(coords) - len(df)
    if n_dropped:
        print(f"[WARN] {n_dropped:,} neurons dropped (no matching super_class).")

    df.rename(columns={"y_coord": "score", "super_class": "target"}, inplace=True)
    print(f"[INFO] Neurons after merge: {len(df):,}")

    # --- 3. Encode target labels ---------------------------------------------
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["target"])
    print(f"[INFO] Unique super-classes: {len(le.classes_):,}")

    # --- 4. Decision Tree on score → super_class -----------------------------
    print(
        f"[INFO] Fitting Decision Tree "
        f"(criterion=entropy, max_leaf_nodes={max_leaves}, "
        f"min_samples_leaf={min_samples_leaf}) …"
    )
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_leaf_nodes=max_leaves,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(df[["score"]], df["target_encoded"])

    # Report split thresholds
    thresholds = [t for t in tree.tree_.threshold if t != -2.0]
    sorted_thresholds = sorted(thresholds, reverse=True)
    print(f"\n[INFO] Split thresholds (score, descending — i.e. level boundary i | i+1):")
    for i, t in enumerate(sorted_thresholds, 1):
        print(f"       Boundary {i} (between level {i} and {i+1}): {t:.6f}")

    # --- 5. Map leaf nodes → ordered integer levels --------------------------
    df["_leaf_id"] = tree.apply(df[["score"]])

    # Level 1 = highest mean score, Level N = lowest
    level_order = (
        df.groupby("_leaf_id")["score"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    leaf_to_level = {leaf: lvl for lvl, leaf in enumerate(level_order, start=1)}
    df["y_level"] = df["_leaf_id"].map(leaf_to_level)

    n_levels = df["y_level"].nunique()
    print(f"\n[INFO] Assigned {n_levels} levels.")

    # --- 6. Per-level statistics ----------------------------------------------
    print("\n[INFO] Per-level statistics:")
    print(f"  {'Level':<8} {'Neurons':>8}  {'Mean score':>12}  {'Entropy':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*10}")
    for lvl in sorted(df["y_level"].unique()):
        mask   = df["y_level"] == lvl
        subset = df[mask]
        probs  = subset["target"].value_counts(normalize=True)
        h      = entropy(probs)
        print(
            f"  {lvl:<8} {len(subset):>8,}  "
            f"{subset['score'].mean():>12.4f}  {h:>10.4f}"
        )

    mean_h = (
        df.groupby("y_level")["target"]
        .apply(lambda x: entropy(x.value_counts(normalize=True)))
        .mean()
    )
    print(f"\n  Global mean entropy: {mean_h:.4f}")

    # --- 7. Save output -------------------------------------------------------
    out = df[["root_id", "x_coord", "score", "y_level"]].copy()
    out.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Saved {len(out):,} rows to '{output_file}' in {elapsed:.1f}s")
    print(out.head())


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Resolve ../data relative to this script's location, regardless of the
    # working directory from which the script is called.
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.normpath(os.path.join(script_dir, "..", "data"))

    parser = argparse.ArgumentParser(
        description="Assign functional levels via Decision Tree on ../data/COORDINATE_XY.csv"
    )
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=5,
        help="Maximum number of leaf nodes in the Decision Tree (default: 5)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per leaf node (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the Decision Tree (default: 42)"
    )
    args = parser.parse_args()

    print(f"[INFO] Data folder: {data_folder}")

    assign_levels(
        data_folder=data_folder,
        max_leaves=args.max_leaves,
        min_samples_leaf=args.min_samples,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()