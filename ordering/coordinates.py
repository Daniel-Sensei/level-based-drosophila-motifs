#!/usr/bin/env python3
"""
coordinates.py
─────────────────────────
Reads  ../data/connections.csv 
and produces  ../data/COORDINATE_XY.csv.

Columns in output:
  root_id   – neuron identifier
  x_coord   – Fiedler vector (2nd eigenvector of the symmetric Laplacian)
  y_coord   – hierarchical ordering score (conjugate-gradient solution of L·y = b)

Usage:
  python coordinates.py
  python coordinates.py --tol 1e-3 --max-iter 200000
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_coordinates(data_folder: str, tol: float = 1e-3, max_iter: int | None = None) -> None:
    start_time = time.time()

    conn_file   = os.path.join(data_folder, "connections.csv")
    output_file = os.path.join(data_folder, "COORDINATE_XY.csv")

    # --- 1. Load data --------------------------------------------------------
    print(f"[INFO] Loading: {conn_file}")
    if not os.path.exists(conn_file):
        print(f"[ERROR] File not found: {conn_file}")
        sys.exit(1)

    df = pd.read_csv(conn_file)
    print(f"[INFO] Loaded {len(df):,} edges")

    required_cols = {"pre_root_id", "post_root_id", "syn_count"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] connections.csv must contain columns: {required_cols}")
        sys.exit(1)

    # --- 2. Node index -------------------------------------------------------
    nodes = np.unique(
        np.concatenate([df["pre_root_id"].unique(), df["post_root_id"].unique()])
    )
    n = len(nodes)
    print(f"[INFO] Unique nodes: {n:,}")

    id_to_idx = {node: idx for idx, node in enumerate(nodes)}
    rows    = df["pre_root_id"].map(id_to_idx).to_numpy()
    cols_   = df["post_root_id"].map(id_to_idx).to_numpy()
    weights = df["syn_count"].to_numpy()

    # --- 3. Sparse matrices --------------------------------------------------
    print("[INFO] Building sparse matrices …")
    W_directed = sp.coo_matrix((weights, (rows, cols_)), shape=(n, n)).tocsc()

    # Symmetric weight matrix
    W = (W_directed + W_directed.transpose()) * 0.5

    # Antisymmetric difference
    Delta = W_directed - W_directed.transpose()

    # Graph Laplacian  L = D - W
    d = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(d)
    L = (D - W).tocsr()

    # Balance / right-hand-side vector  b = (W ⊙ Δ) · 1
    WD = W.multiply(Delta)
    b  = np.array(WD.sum(axis=1)).flatten()

    # --- 4. Conjugate Gradient  L·y = b  (y-coord) --------------------------
    if max_iter is None:
        max_iter = n

    print(f"[INFO] Running Conjugate Gradient (tol={tol}, max_iter={max_iter}) …")

    np.random.seed(42)
    y = np.random.rand(n)
    y -= y.mean()

    r         = b - L.dot(y)
    d_vec     = r.copy()
    delta_new = r.dot(r)
    delta_0   = delta_new

    converged = False
    with tqdm(total=max_iter, desc="Conjugate Gradient", ncols=100) as pbar:
        for i in range(max_iter):
            if delta_new < tol ** 2 * delta_0:
                print(f"\n[INFO] Converged after {i} iterations")
                converged = True
                break
            q         = L.dot(d_vec)
            alpha     = delta_new / d_vec.dot(q)
            y        += alpha * d_vec
            r         = b - L.dot(y) if i % 50 == 0 else r - alpha * q
            delta_old = delta_new
            delta_new = r.dot(r)
            beta      = delta_new / delta_old
            d_vec     = r + beta * d_vec
            y        -= y.mean()
            pbar.update(1)

    if not converged:
        print("[WARN] Conjugate Gradient did not fully converge.")

    # --- 5. Fiedler vector  (x-coord) ----------------------------------------
    print("[INFO] Computing Fiedler vector (2nd smallest eigenvector) …")
    with tqdm(total=1, desc="Fiedler Vector", ncols=100) as pbar:
        vals, vecs = spla.eigsh(L, k=2, which="SM", tol=1e-3)
        x = vecs[:, 1]
        pbar.update(1)

    # --- 6. Save output -------------------------------------------------------
    df_coords = pd.DataFrame({
        "root_id": nodes,
        "x_coord": x,
        "y_coord": y,
    })
    df_coords.to_csv(output_file, index=False)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Saved {len(df_coords):,} rows to '{output_file}' in {elapsed:.1f}s")
    print(df_coords.head())


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Resolve ../data relative to this script's location, regardless of the
    # working directory from which the script is called.
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.normpath(os.path.join(script_dir, "..", "data"))

    parser = argparse.ArgumentParser(
        description="Compute XY coordinates (Fiedler + CG) from ../data/connections.csv"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for the Conjugate Gradient solver (default: 1e-3)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Maximum CG iterations (default: number of nodes)"
    )
    args = parser.parse_args()

    print(f"[INFO] Data folder: {data_folder}")

    compute_coordinates(
        data_folder=data_folder,
        tol=args.tol,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
