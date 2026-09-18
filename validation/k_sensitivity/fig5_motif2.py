#!/usr/bin/env python3
"""
Figure 5 motif sensitivity to synaptic-count threshold K.

For each K in 0..10, an edge is kept iff:
    syn_count > K

Therefore:
    K=0  -> current implementation (syn_count > 0)
    K=5  -> requested review threshold (syn_count > 5)
    K=10 -> syn_count > 10

The neuron levels/classification are kept FIXED. Only the connectivity
matrix used for motif counting changes.

Motif:
    sensory L1 -> central L2 -> central L3 -> central L2

The count uses the same flow-vector semantics as the paper-aligned
Figure 5 validation:
    ones @ M01 @ M12 @ M23
and then sum of the final flow vector.

Outputs:
    fig5_motif2_count_vs_K.csv
    fig5_motif2_count_vs_K.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


FIG_5_MOTIF_2 = {
    "name": "fig_5_motif_2",
    "blocks": [
        ("sensory", 1),
        ("central", 2),
        ("central", 3),
        ("central", 2),
    ],
    "edges": [(0, 1), (1, 2), (2, 3)],
}


def load_metadata(levels_file, classification_file):
    df_levels = pd.read_csv(levels_file)
    df_class = pd.read_csv(classification_file)

    required_levels = {"root_id", "y_level"}
    required_class = {"root_id", "super_class"}

    missing = required_levels - set(df_levels.columns)
    if missing:
        raise ValueError(f"Missing columns in levels: {sorted(missing)}")

    missing = required_class - set(df_class.columns)
    if missing:
        raise ValueError(
            f"Missing columns in classification: {sorted(missing)}"
        )

    # Same neuron universe convention used by motif_enumeration.py.
    all_neurons = sorted(
        set(df_levels["root_id"]) | set(df_class["root_id"])
    )

    neuron_to_idx = {nid: i for i, nid in enumerate(all_neurons)}
    idx_to_neuron = {
        i: nid for nid, i in neuron_to_idx.items()
    }

    level_map = (
        df_levels
        .set_index("root_id")["y_level"]
        .to_dict()
    )

    class_map = (
        df_class
        .set_index("root_id")["super_class"]
        .dropna()
        .to_dict()
    )

    groups = {}

    for nid in all_neurons:
        if nid not in level_map or nid not in class_map:
            continue

        key = (class_map[nid], level_map[nid])
        groups.setdefault(key, [])
        groups[key].append(neuron_to_idx[nid])

    groups = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in groups.items()
    }

    return neuron_to_idx, idx_to_neuron, groups


def load_connections(connections_file):
    df_conn = pd.read_csv(connections_file)

    required = {"pre_root_id", "post_root_id", "syn_count"}
    missing = required - set(df_conn.columns)
    if missing:
        raise ValueError(
            f"Missing columns in connections: {sorted(missing)}"
        )

    return df_conn


def build_adjacency(df_conn, neuron_to_idx, threshold):
    """
    Keep exactly the same binary-edge semantics as compute.py, but make
    the synaptic threshold configurable.

        current code: syn_count > 0
        this script : syn_count > threshold
    """
    df_edges = df_conn[
        df_conn["syn_count"] > threshold
    ][["pre_root_id", "post_root_id"]].drop_duplicates()

    s_pre = df_edges["pre_root_id"].map(neuron_to_idx)
    s_post = df_edges["post_root_id"].map(neuron_to_idx)

    valid = s_pre.notna() & s_post.notna()

    s_pre = s_pre[valid].astype(int).to_numpy()
    s_post = s_post[valid].astype(int).to_numpy()

    adjacency = sparse.csr_matrix(
        (
            np.ones(len(s_pre), dtype=np.uint8),
            (s_pre, s_post),
        ),
        shape=(len(neuron_to_idx), len(neuron_to_idx)),
        dtype=np.uint8,
    )

    return adjacency


def validate_motif(motif, groups):
    blocks = motif["blocks"]
    expected_edges = [
        (i, i + 1)
        for i in range(len(blocks) - 1)
    ]

    if motif["edges"] != expected_edges:
        raise ValueError(
            f"Motif edges must be {expected_edges}; "
            f"got {motif['edges']}"
        )

    for pos, block in enumerate(blocks):
        if block not in groups:
            raise ValueError(
                f"Motif position {pos} uses missing block {block}"
            )


def count_occurrences(motif, groups, adjacency):
    """
    Exact paper/compute.py flow-vector count for one fixed motif sequence.
    """
    validate_motif(motif, groups)

    blocks = motif["blocks"]
    matrices = []

    for i in range(len(blocks) - 1):
        src_indices = groups[blocks[i]]
        dst_indices = groups[blocks[i + 1]]

        matrix = adjacency[src_indices, :][:, dst_indices].tocsr()
        matrices.append(matrix)

    counts = np.ones(
        len(groups[blocks[0]]),
        dtype=np.float64,
    )

    for matrix in matrices:
        counts = np.asarray(counts @ matrix).ravel()

        if counts.sum() == 0:
            return 0

    return int(counts.sum())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Figure 5 Motif 2 count for K=0..10, "
            "where an edge is retained when syn_count > K."
        )
    )

    parser.add_argument(
        "--connections",
        default="../../data/connections.csv",
        help="Path to connections.csv",
    )
    parser.add_argument(
        "--levels",
        default="../../data/COORDINATE_XY_with_levels_tree.csv",
        help="Path to COORDINATE_XY_with_levels_tree.csv",
    )
    parser.add_argument(
        "--classification",
        default="../../data/classification.csv",
        help="Path to classification.csv",
    )
    parser.add_argument(
        "--outdir",
        default="./results_fig5_motif2_k_sensitivity",
        help="Output directory",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=0,
        help="Minimum K threshold (default: 0)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum K threshold (default: 10)",
    )

    args = parser.parse_args()

    if args.k_min < 0 or args.k_max < args.k_min:
        raise ValueError("Invalid K range.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FIGURE 5 — MOTIF 2 SENSITIVITY TO SYNAPTIC THRESHOLD K")
    print("=" * 78)

    print("\n[1] Loading data...")
    df_conn = load_connections(args.connections)
    neuron_to_idx, idx_to_neuron, groups = load_metadata(
        args.levels,
        args.classification,
    )

    print(f"Total neurons: {len(neuron_to_idx):,}")
    print(f"Connection rows in CSV: {len(df_conn):,}")

    print("\nMotif:")
    print(
        "  "
        + " -> ".join(
            f"{sc}, L{level}"
            for sc, level in FIG_5_MOTIF_2["blocks"]
        )
    )

    print("\nImportant:")
    print("  K=0 means syn_count > 0, i.e. the current implementation.")
    print("  K=5 means syn_count > 5.")
    print("  Levels and classification are NOT recomputed for each K.")
    print("  Only the adjacency/connectivity threshold changes.")

    results = []

    print("\n[2] Computing counts...")
    print("-" * 78)
    print(f"{'K':>4} {'condition':>18} {'edges':>15} {'count':>20}")
    print("-" * 78)

    for k in range(args.k_min, args.k_max + 1):
        adjacency = build_adjacency(
            df_conn,
            neuron_to_idx,
            threshold=k,
        )

        count = count_occurrences(
            FIG_5_MOTIF_2,
            groups,
            adjacency,
        )

        results.append({
            "K": k,
            "condition": f"syn_count > {k}",
            "edges": adjacency.nnz,
            "motif": FIG_5_MOTIF_2["name"],
            "count": count,
        })

        print(
            f"{k:>4} "
            f"{'syn_count > ' + str(k):>18} "
            f"{adjacency.nnz:>15,} "
            f"{count:>20,}"
        )

    results_df = pd.DataFrame(results)

    csv_path = outdir / "fig5_motif2_count_vs_K.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n[3] Creating plot...")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        results_df["K"],
        results_df["count"],
        marker="o",
    )

    ax.set_xlabel("Synaptic threshold K")
    ax.set_ylabel("Motif occurrence count")
    ax.set_title(
        "Figure 5 — Motif 2 occurrence count vs synaptic threshold K"
    )
    ax.set_xticks(results_df["K"])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = outdir / "fig5_motif2_count_vs_K.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"\n[OUTPUT] CSV : {csv_path.resolve()}")
    print(f"[OUTPUT] Plot: {plot_path.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
