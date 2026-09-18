#!/usr/bin/env python3
"""
motif_enumeration.py

Neuron-level validation of the motif-counting semantics implemented in
computation/compute.py.

IMPORTANT
---------
The paper/code represents a motif as a SEQUENCE OF BLOCKS. If a block
appears twice in the sequence, those are two traversal positions and they
do NOT have to contain the same neuron.

For Figure 5 the paper motifs are therefore:

    motif 2:
        sensory L1 -> central L2 -> central L3 -> central L2

    motif 3:
        central L1 -> central L2 -> central L3 -> central L2

The corresponding neuron-level occurrence is:

    n0 -> n1 -> n2 -> n3

where:
    n0 belongs to block 0
    n1 belongs to block 1
    n2 belongs to block 2
    n3 belongs to block 3

and n1 and n3 may be DIFFERENT neurons, even though both belong to
central L2.

This is the crucial distinction from treating the motif as a 3-node
graph with a closing edge (2 -> 1).

The count is exactly the same algebra used in computation/compute.py:

    ones @ M01 @ M12 @ M23

followed by the L1 sum.

We additionally compute the sets of neuron IDs that occur at each
sequence position in at least one complete occurrence, without storing
all individual occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


Block = Tuple[str, int]
Edge = Tuple[int, int]


@dataclass
class MotifResult:
    name: str
    blocks: List[Block]
    edges: List[Edge]

    # Number of neuron-level occurrences.
    occurrence_count: int

    # Root IDs participating at each POSITION in the motif sequence.
    neurons_by_position: Dict[int, Set[int]]

    # Union of root IDs appearing at any position.
    neurons: Set[int]

    @property
    def neuron_count(self) -> int:
        return len(self.neurons)


def load_data(
    connections_file: str,
    levels_file: str,
    classification_file: str,
):
    """
    Load the same inputs and adjacency semantics as computation/compute.py.

    Returns
    -------
    neuron_to_idx
        root_id -> global matrix index
    idx_to_neuron
        global matrix index -> root_id
    groups
        (super_class, level) -> list/array of global matrix indices
    adjacency
        sparse binary global adjacency matrix
    """
    print("[INFO] Reading CSV files...")
    df_conn = pd.read_csv(connections_file)
    df_levels = pd.read_csv(levels_file)
    df_class = pd.read_csv(classification_file)

    required_conn = {"pre_root_id", "post_root_id", "syn_count"}
    required_levels = {"root_id", "y_level"}
    required_class = {"root_id", "super_class"}

    missing = required_conn - set(df_conn.columns)
    if missing:
        raise ValueError(
            f"connections file missing columns: {sorted(missing)}"
        )

    missing = required_levels - set(df_levels.columns)
    if missing:
        raise ValueError(
            f"levels file missing columns: {sorted(missing)}"
        )

    missing = required_class - set(df_class.columns)
    if missing:
        raise ValueError(
            f"classification file missing columns: {sorted(missing)}"
        )

    # This follows compute.py.
    all_neurons = sorted(
        list(set(df_levels.root_id) | set(df_class.root_id))
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

    groups: Dict[Block, np.ndarray] = {}

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

    # Same edge semantics as compute.py:
    #   syn_count > 0
    #   duplicate directed pairs removed
    df_edges = (
        df_conn[df_conn["syn_count"] > 0]
        [["pre_root_id", "post_root_id"]]
        .drop_duplicates()
    )

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
        shape=(len(all_neurons), len(all_neurons)),
        dtype=np.uint8,
    )

    return neuron_to_idx, idx_to_neuron, groups, adjacency


def validate_motif(motif: dict, groups: dict) -> None:
    blocks = motif["blocks"]
    edges = motif["edges"]

    if len(blocks) < 2:
        raise ValueError("A motif must contain at least two positions.")

    # For this validation script we intentionally require a sequence:
    # 0 -> 1 -> 2 -> ... -> k-1.
    expected_edges = [
        (i, i + 1)
        for i in range(len(blocks) - 1)
    ]

    if list(edges) != expected_edges:
        raise ValueError(
            "The motif must be represented as a block sequence with "
            f"edges {expected_edges}. Got {edges}."
        )

    for pos, block in enumerate(blocks):
        if block not in groups:
            raise ValueError(
                f"Motif position {pos} uses block {block}, "
                "but this block does not exist in the data."
            )


def build_sequence_matrices(
    motif: dict,
    groups: dict,
    adjacency: sparse.csr_matrix,
) -> List[sparse.csr_matrix]:
    """
    Build M_i,i+1 for the motif sequence.

    If the same block occurs at different positions, each position gets
    its own matrix dimension. For Figure 5:

        M01: sensory L1  -> central L2
        M12: central L2  -> central L3
        M23: central L3  -> central L2

    Notice that M23 targets a NEW traversal position in central L2.
    It is not a constraint that this target neuron equals the source
    central-L2 neuron at position 1.
    """
    validate_motif(motif, groups)

    blocks = motif["blocks"]
    matrices = []

    for i in range(len(blocks) - 1):
        src_indices = groups[blocks[i]]
        dst_indices = groups[blocks[i + 1]]

        submat = adjacency[src_indices, :][:, dst_indices].tocsr()
        submat.data[:] = 1
        submat.eliminate_zeros()

        matrices.append(submat)

    return matrices


def count_occurrences(
    motif: dict,
    groups: dict,
    adjacency: sparse.csr_matrix,
) -> int:
    """
    Reproduce the paper/compute.py flow-vector count for one fixed
    motif sequence.

    For Figure 5:

        v0 = ones
        v1 = v0 @ M01
        v2 = v1 @ M12
        v3 = v2 @ M23

        count = sum(v3)

    This counts distinct neuron-level paths n0 -> n1 -> n2 -> n3.
    In particular, n1 and n3 are allowed to be different neurons from
    the same (central, L2) block.
    """
    matrices = build_sequence_matrices(
        motif, groups, adjacency
    )

    counts = np.ones(
        len(groups[motif["blocks"][0]]),
        dtype=np.float64,
    )

    for mat in matrices:
        counts = counts @ mat

        if np.sum(counts) == 0:
            return 0

    return int(np.sum(counts))


def participating_neurons(
    motif: dict,
    groups: dict,
    adjacency: sparse.csr_matrix,
) -> Dict[int, Set[int]]:
    """
    Find all neurons that participate in at least one COMPLETE occurrence.

    No individual occurrences are stored.

    For a simple sequence:

        B0 -> B1 -> B2 -> B3

    a neuron at position i is retained iff it belongs to at least one
    complete path through all positions.

    Because the motif is a linear sequence, this can be computed exactly
    by forward/backward boolean reachability.
    """
    validate_motif(motif, groups)

    matrices = build_sequence_matrices(
        motif, groups, adjacency
    )

    n_positions = len(motif["blocks"])

    # Forward reachability:
    # forward[i][j] = neuron j at position i can be reached from at least
    # one neuron at position 0.
    forward = [
        np.zeros(len(groups[motif["blocks"][i]]), dtype=bool)
        for i in range(n_positions)
    ]

    forward[0][:] = True

    for i, mat in enumerate(matrices):
        supported = np.asarray(
            mat[forward[i], :].getnnz(axis=0)
        ).ravel() > 0

        forward[i + 1] = supported

    # Backward reachability:
    # backward[i][j] = neuron j at position i can reach at least one
    # neuron at the final position.
    backward = [
        np.zeros(len(groups[motif["blocks"][i]]), dtype=bool)
        for i in range(n_positions)
    ]

    backward[-1][:] = True

    for i in range(n_positions - 2, -1, -1):
        mat = matrices[i]

        supported = np.asarray(
            mat[:, backward[i + 1]].getnnz(axis=1)
        ).ravel() > 0

        backward[i] = supported

    # A neuron participates in a COMPLETE occurrence iff it is reachable
    # from the beginning AND can reach the end.
    neurons_by_position = {}

    for i, block in enumerate(motif["blocks"]):
        complete_mask = forward[i] & backward[i]

        global_indices = groups[block][complete_mask]

        neurons_by_position[i] = {
            int(idx)
            for idx in global_indices
        }

    return neurons_by_position


def analyze_motif(
    motif: dict,
    groups: dict,
    adjacency: sparse.csr_matrix,
    idx_to_neuron: dict,
) -> MotifResult:
    """
    Run both:
        1. exact flow-vector occurrence counting
        2. exact participating-neuron analysis
    """
    validate_motif(motif, groups)

    count = count_occurrences(
        motif,
        groups,
        adjacency,
    )

    participating_indices = participating_neurons(
        motif,
        groups,
        adjacency,
    )

    neurons_by_position = {
        pos: {
            idx_to_neuron[int(idx)]
            for idx in indices
        }
        for pos, indices in participating_indices.items()
    }

    all_neurons = set()

    for values in neurons_by_position.values():
        all_neurons.update(values)

    return MotifResult(
        name=motif["name"],
        blocks=list(motif["blocks"]),
        edges=list(motif["edges"]),
        occurrence_count=count,
        neurons_by_position=neurons_by_position,
        neurons=all_neurons,
    )


def enumerate_occurrences(
    motif: dict,
    groups: dict,
    adjacency: sparse.csr_matrix,
    idx_to_neuron: dict,
    max_occurrences: int | None = None,
) -> Iterator[Tuple[int, ...]]:
    """
    Optional explicit DFS over INDIVIDUAL NEURONS.

    This is ONLY for validating small datasets or printing examples.

    On the real connectome, do NOT use this to reproduce millions of
    occurrences unless you deliberately want to enumerate them all.

    The semantics are exactly:

        position 0 -> position 1 -> ... -> position k

    and repeated blocks represent independent neuron positions.
    """
    validate_motif(motif, groups)

    matrices = build_sequence_matrices(
        motif, groups, adjacency
    )

    n_positions = len(motif["blocks"])
    assignment = [None] * n_positions

    yielded = 0

    # Convert matrices to CSR for fast row access.
    matrices = [m.tocsr() for m in matrices]

    def dfs(position: int):
        nonlocal yielded

        if (
            max_occurrences is not None
            and yielded >= max_occurrences
        ):
            return

        if position == n_positions:
            yielded += 1

            yield tuple(
                idx_to_neuron[
                    int(
                        groups[motif["blocks"][i]][
                            assignment[i]
                        ]
                    )
                ]
                for i in range(n_positions)
            )
            return

        if position == 0:
            candidates = range(
                len(groups[motif["blocks"][0]])
            )
        else:
            prev_local_idx = assignment[position - 1]
            row = matrices[position - 1].getrow(
                prev_local_idx
            )

            candidates = row.indices

        for local_idx in candidates:
            assignment[position] = int(local_idx)
            yield from dfs(position + 1)

        assignment[position] = None

    yield from dfs(0)


def print_result(result: MotifResult) -> None:
    print()
    print("=" * 72)
    print(result.name)
    print("=" * 72)

    print(
        f"Occurrence count : "
        f"{result.occurrence_count:,}"
    )
    print(
        f"Unique neurons   : "
        f"{result.neuron_count:,}"
    )

    for pos, block in enumerate(result.blocks):
        print(
            f"Position {pos}: "
            f"{block[0]} L{block[1]} -> "
            f"{len(result.neurons_by_position[pos]):,} "
            f"participating neurons"
        )


if __name__ == "__main__":
    print(
        "This module is intended to be imported by "
        "jaccard.py."
    )
