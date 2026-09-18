#!/usr/bin/env python3
"""
motif_jaccard.py

Jaccard similarity between the sets of neuron IDs participating in two
motifs.

For motifs A and B:
    J(A,B) = |A ∩ B| / |A ∪ B|

where A and B are the UNION of all neuron IDs appearing in at least one
complete occurrence of the respective motif.
"""

from __future__ import annotations

from typing import Iterable, Set


def jaccard_similarity(
    set_a: Iterable[int],
    set_b: Iterable[int],
) -> float:
    """
    Compute standard Jaccard similarity.
    """
    a: Set[int] = set(set_a)
    b: Set[int] = set(set_b)

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def compare_motifs(motif_a, motif_b) -> dict:
    """
    Compare two MotifResult objects returned by motif_enumeration.analyze_motif.
    """
    neurons_a = set(motif_a.neurons)
    neurons_b = set(motif_b.neurons)

    intersection = neurons_a & neurons_b
    union = neurons_a | neurons_b

    return {
        "motif_a": motif_a.name,
        "motif_b": motif_b.name,
        "occurrences_a": motif_a.occurrence_count,
        "occurrences_b": motif_b.occurrence_count,
        "neurons_a": len(neurons_a),
        "neurons_b": len(neurons_b),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": (
            len(intersection) / len(union)
            if union
            else 1.0
        ),
        "intersection_neurons": sorted(intersection),
        "union_neurons": sorted(union),
    }


def save_comparison_csv(comparison: dict, output_file: str) -> None:
    """
    Save the scalar Jaccard result to CSV.
    """
    import pandas as pd

    row = {
        key: value
        for key, value in comparison.items()
        if key not in {"intersection_neurons", "union_neurons"}
    }

    pd.DataFrame([row]).to_csv(output_file, index=False)


def save_neuron_sets_csv(comparison: dict, output_file: str) -> None:
    """
    Save neuron-level membership information.

    Columns:
        neuron_id
        in_motif_a
        in_motif_b
        in_intersection
    """
    a = set(comparison["intersection_neurons"])
    union = set(comparison["union_neurons"])

    # Recover B from intersection/union is not possible, so this helper is
    # intentionally kept simple: it is normally called from
    # validate_jaccard.py, which has direct access to both motif sets.
    raise RuntimeError(
        "Use save_detailed_neuron_sets() with the two MotifResult objects."
    )


def save_detailed_neuron_sets(
    motif_a,
    motif_b,
    output_file: str,
) -> None:
    import pandas as pd

    a = set(motif_a.neurons)
    b = set(motif_b.neurons)

    rows = [
        {
            "neuron_id": neuron,
            "in_motif_a": neuron in a,
            "in_motif_b": neuron in b,
            "in_intersection": neuron in (a & b),
        }
        for neuron in sorted(a | b)
    ]

    pd.DataFrame(rows).to_csv(output_file, index=False)
