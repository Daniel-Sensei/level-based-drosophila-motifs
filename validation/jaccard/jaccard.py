#!/usr/bin/env python3
"""
jaccard.py

Analyzes and counts occurrences for the motifs in Figures 5, 6, 9, and 10.
Computes the Jaccard similarity for the specific motif pairs within each figure
and saves the detailed neuron sets and a summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from motif_enumeration import (
    load_data,
    analyze_motif,
    print_result,
)
from motif_jaccard import (
    compare_motifs,
    save_detailed_neuron_sets,
)

# ---------------------------------------------------------------------------
# MOTIF CONFIGURATIONS
# ---------------------------------------------------------------------------

# --- FIGURE 5 ---
FIG_5_MOTIF_2 = {
    "name": "fig_5_motif_2",
    "blocks": [("sensory", 1), ("central", 2), ("central", 3), ("central", 2)],
    "edges": [(0, 1), (1, 2), (2, 3)],
}

FIG_5_MOTIF_3 = {
    "name": "fig_5_motif_3",
    "blocks": [("central", 1), ("central", 2), ("central", 3), ("central", 2)],
    "edges": [(0, 1), (1, 2), (2, 3)],
}

# --- FIGURE 6 ---
FIG_6_MOTIF_1 = {
    "name": "fig_6_motif_1",
    "blocks": [
        ("optic", 2), ("optic", 3), ("optic", 4), ("optic", 5), ("optic", 4)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
}

FIG_6_MOTIF_4 = {
    "name": "fig_6_motif_4",
    "blocks": [
        ("optic", 2), ("optic", 3), ("optic", 4), ("visual_centrifugal", 5), ("optic", 4)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
}

# --- FIGURE 9 ---
FIG_9_MOTIF_1 = {
    "name": "fig_9_motif_1",
    "blocks": [
        ("sensory", 1), ("central", 1), ("central", 2), ("ascending", 3),
        ("central", 3), ("visual_centrifugal", 2), ("optic", 1), ("optic", 2),
        ("optic", 3), ("optic", 2)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
}

FIG_9_MOTIF_2 = {
    "name": "fig_9_motif_2",
    "blocks": [
        ("sensory", 1), ("central", 1), ("central", 2), ("central", 3),
        ("visual_centrifugal", 2), ("optic", 3), ("visual_projection", 3), ("optic", 2),
        ("optic", 1), ("optic", 2)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
}

# --- FIGURE 10 ---
FIG_10_MOTIF_1 = {
    "name": "fig_10_motif_1",
    "blocks": [
        ("optic", 2), ("optic", 3), ("visual_projection", 4), ("visual_centrifugal", 4),
        ("visual_projection", 3), ("central", 4), ("central", 3), ("central", 2),
        ("visual_centrifugal", 2), ("optic", 2)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
}

FIG_10_MOTIF_2 = {
    "name": "fig_10_motif_2",
    "blocks": [
        ("optic", 2), ("optic", 3), ("optic", 4), ("visual_projection", 4),
        ("visual_centrifugal", 4), ("visual_projection", 3), ("central", 3),
        ("central", 2), ("visual_centrifugal", 2), ("optic", 2)
    ],
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
}

# --- PAIRS TO ANALYZE AND COMPARE ---
PAIRS_TO_COMPARE = [
    {
        "figure": "Figure 5",
        "motif_a": FIG_5_MOTIF_2,
        "motif_b": FIG_5_MOTIF_3,
    },
    {
        "figure": "Figure 6",
        "motif_a": FIG_6_MOTIF_1,
        "motif_b": FIG_6_MOTIF_4,
    },
    {
        "figure": "Figure 9",
        "motif_a": FIG_9_MOTIF_1,
        "motif_b": FIG_9_MOTIF_2,
    },
    {
        "figure": "Figure 10",
        "motif_a": FIG_10_MOTIF_1,
        "motif_b": FIG_10_MOTIF_2,
    },
]

def main():
    parser = argparse.ArgumentParser(
        description="Compute occurrence counts and Jaccard similarity for motifs in Figures 5, 6, 9, and 10."
    )
    parser.add_argument("--connections", default="../../data/connections.csv", help="Path to connections.csv")
    parser.add_argument("--levels", default="../../data/COORDINATE_XY_with_levels_tree.csv", help="Path to COORDINATE_XY_with_levels_tree.csv")
    parser.add_argument("--classification", default="../../data/classification.csv", help="Path to classification.csv")
    parser.add_argument("--outdir", default="./jaccard_results", help="Directory for output CSV files")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MOTIF ANALYSIS & JACCARD SIMILARITY (FIGURES 5, 6, 9, 10)")
    print("=" * 72)

    print("\n[1] Loading connectome data...")
    neuron_to_idx, idx_to_neuron, groups, adjacency = load_data(
        args.connections,
        args.levels,
        args.classification,
    )
    
    print(f"Total neurons in mappings : {len(neuron_to_idx):,}")
    print(f"Adjacency shape           : {adjacency.shape}")
    print(f"Adjacency nonzeros        : {adjacency.nnz:,}")

    summary_results = []

    print("\n[2] Analyzing motifs and computing Jaccard similarities...")
    
    for pair in PAIRS_TO_COMPARE:
        figure = pair["figure"]
        motif_a_def = pair["motif_a"]
        motif_b_def = pair["motif_b"]
        
        print(f"\n{'=' * 40}")
        print(f"Processing {figure}: {motif_a_def['name']} vs {motif_b_def['name']}")
        print(f"{'=' * 40}")

        try:
            # Analyze Motif A
            result_a = analyze_motif(motif_a_def, groups, adjacency, idx_to_neuron)
            print_result(result_a)
            
            # Analyze Motif B
            result_b = analyze_motif(motif_b_def, groups, adjacency, idx_to_neuron)
            print_result(result_b)
            
            # Compare and calculate Jaccard
            comparison = compare_motifs(result_a, result_b)
            
            print("\n--- Jaccard Result ---")
            print(f"Intersection : {comparison['intersection']:,}")
            print(f"Union        : {comparison['union']:,}")
            print(f"Jaccard      : {comparison['jaccard']:.10f}")

            # Append core scalar values to summary list
            summary_row = {
                "Figure": figure,
                "Motif_A": comparison["motif_a"],
                "Motif_B": comparison["motif_b"],
                "Occurrences_A": comparison["occurrences_a"],
                "Occurrences_B": comparison["occurrences_b"],
                "Unique_Neurons_A": comparison["neurons_a"],
                "Unique_Neurons_B": comparison["neurons_b"],
                "Intersection": comparison["intersection"],
                "Union": comparison["union"],
                "Jaccard_Similarity": comparison["jaccard"]
            }
            summary_results.append(summary_row)
            
            # Save detailed neuron sets for this pair
            neuron_csv = outdir / f"{figure.lower().replace(' ', '_')}_neurons.csv"
            save_detailed_neuron_sets(result_a, result_b, str(neuron_csv))
            print(f"[OUTPUT] Detailed neurons saved to: {neuron_csv.name}")

        except ValueError as e:
            print(f"\n[ERROR] Failed to analyze {figure}: {e}")

    print("\n[3] Saving summary CSV...")
    summary_csv = outdir / "jaccard_summary.csv"
    pd.DataFrame(summary_results).to_csv(summary_csv, index=False)
    
    print(f"\n[OUTPUT] Master summary saved to: {summary_csv.resolve()}")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()