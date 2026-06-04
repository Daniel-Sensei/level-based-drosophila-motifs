#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
import time
import sys
from scipy import sparse
from collections import defaultdict
import multiprocessing
from functools import partial

# --- GLOBAL SETTINGS ---
# Threshold for when to convert a sparse matrix to dense for faster multiplication
DENSE_THRESHOLD = 250000 

# Globals for multiprocessing (Copy-on-Write optimization)
G_NEIGHBOR_MAP = None
G_GROUPS = None
G_NH = None
G_TARGET_LEVELS = None

def get_neighbor_map():
    return G_NEIGHBOR_MAP

def check_nh_constraint(path_keys, nh):
    """
    Checks if the path violates the NH (Number of Hops/Classes per level) constraint.
    """
    if nh is None: return True
    level_classes = defaultdict(set)
    for (cls, lev) in path_keys:
        level_classes[lev].add(cls)
        if len(level_classes[lev]) > nh:
            return False
    return True

def worker_process_start_node(start_node_key):
    """
    This function runs inside a worker process.
    It performs the DFS starting from a specific node.
    """
    # Access global read-only data
    neighbor_map = G_NEIGHBOR_MAP
    groups = G_GROUPS
    nh = G_NH
    target_levels = G_TARGET_LEVELS
    
    local_results = []
    
    # Initial setup for this start node
    n_start = len(groups[start_node_key])
    if n_start == 0: return []
    
    initial_counts = np.ones(n_start, dtype=np.float64)
    
    # Stack for Iterative DFS 
    # Stack items: (current_key, path_keys, current_counts, levels_covered_set)
    stack = [ (start_node_key, [start_node_key], initial_counts, {start_node_key[1]}) ]
    
    while stack:
        curr_key, path, counts, covered = stack.pop()
        
        # 1. Check Motif Completion
        if covered == target_levels:
            total = np.sum(counts)
            if total > 0:
                row = {
                    'motif_count': int(total), 
                    'chain_length': len(path),
                    'structure_str': " -> ".join([f"{c}L{l}" for c,l in path]),
                    'path_tuple': tuple(path) # Helper for dict creation later
                }
                local_results.append(row)
        
        # 2. Explore Neighbors
        # neighbors is a list of (next_key, submatrix)
        neighbors = neighbor_map.get(curr_key, [])
        
        for next_key, submat in neighbors:
            next_lev = next_key[1]
            
            # --- Cycle Handling ---
            if next_key in path:
                # Potential cycle closure
                cycle_path = path + [next_key]
                if check_nh_constraint(cycle_path, nh):
                    # Matmul
                    next_counts = counts @ submat 
                    total = np.sum(next_counts)
                    
                    if total > 0:
                        new_covered = covered | {next_lev}
                        if new_covered == target_levels:
                            # Found cycle motif
                            row = {
                                'motif_count': int(total), 
                                'chain_length': len(cycle_path),
                                'structure_str': " -> ".join([f"{c}L{l}" for c,l in cycle_path]),
                                'path_tuple': tuple(cycle_path)
                            }
                            local_results.append(row)
                continue # Don't recurse into closed cycle

            # --- New Node Step ---
            new_path = path + [next_key]
            
            # NH Constraint
            if not check_nh_constraint(new_path, nh):
                continue
            
            # Math: Flow Calculation
            # submat can be sparse (csr) or dense (numpy array). @ handles both.
            next_counts = counts @ submat
            
            # Pruning: Dead flow
            if np.sum(next_counts) == 0:
                continue
                
            new_covered = covered | {next_lev}
            
            # Push to stack
            stack.append( (next_key, new_path, next_counts, new_covered) )
            
    return local_results

def compute_metagraph_topology(NL_WINDOW, out_folder, nh=None, max_jump=1, no_self_loops=False,
                               conn_file="../data/connections.csv",
                               levels_file="../data/COORDINATE_XY_with_levels_tree.csv",
                               class_file="../data/classification.csv",
                               n_cores=1):

    global G_NEIGHBOR_MAP, G_GROUPS, G_NH, G_TARGET_LEVELS
    
    TARGET_LEVELS_SET = set(NL_WINDOW)
    MIN_LEVEL = min(NL_WINDOW)
    G_TARGET_LEVELS = TARGET_LEVELS_SET
    G_NH = nh
    
    os.makedirs(out_folder, exist_ok=True)
    start_time = time.time()

    print(f"--- Processing Window: {NL_WINDOW} ---")
    print(f"[PARAMS] NH={nh}, MaxJump={max_jump}, NoSelfLoops={no_self_loops}, Cores={n_cores}")
    print(f"[MEMORY] Using Hybrid Dense Matrices (Threshold={DENSE_THRESHOLD} elements)")

    # --- 1. Load Data ---
    print(f"[INFO] Loading CSV files...")
    try:
        df_conn = pd.read_csv(conn_file)
        df_levels = pd.read_csv(levels_file)
        df_class = pd.read_csv(class_file)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # --- 2. Mappings ---
    all_neurons = sorted(list(set(df_levels.root_id) | set(df_class.root_id)))
    neuron_to_idx = {nid: i for i, nid in enumerate(all_neurons)}
    n_total = len(all_neurons)

    level_map = df_levels.set_index('root_id')['y_level'].to_dict()
    class_map = df_class.set_index('root_id')['super_class'].dropna().to_dict()

    valid_neurons = [n for n in all_neurons if n in level_map and n in class_map]
    groups = defaultdict(list)
    
    for n in valid_neurons:
        lev = level_map[n]
        if lev in TARGET_LEVELS_SET:
            cls = class_map[n]
            idx = neuron_to_idx[n]
            groups[(cls, lev)].append(idx)
            
    G_GROUPS = groups
    group_keys = sorted(list(groups.keys()))
    print(f"[INFO] Found {len(group_keys)} valid (Class, Level) groups.")

    # --- 3. Build Adjacency ---
    print(f"[INFO] Building adjacency matrix...")
    df_edges = df_conn[df_conn['syn_count'] > 0][['pre_root_id', 'post_root_id']]
    df_edges = df_edges.drop_duplicates()
    s_pre = df_edges['pre_root_id'].map(neuron_to_idx).dropna().astype(int)
    s_post = df_edges['post_root_id'].map(neuron_to_idx).dropna().astype(int)
    data = np.ones(len(s_pre)) 
    adj_matrix = sparse.csr_matrix((data, (s_pre, s_post)), shape=(n_total, n_total))
    
    # --- 4. Hybrid Pre-computation ---
    print(f"[INFO] Pre-computing Hybrid Connectivity Graph...")
    neighbor_map = defaultdict(list)
    
    precalc_count = 0
    dense_count = 0
    
    for src_key in group_keys:
        src_lev = src_key[1]
        src_indices = groups[src_key]
        row_submat = adj_matrix[src_indices, :]
        
        for dst_key in group_keys:
            dst_lev = dst_key[1]
            
            # Topology filters
            if abs(src_lev - dst_lev) > max_jump: continue
            if no_self_loops and src_key == dst_key: continue
            
            dst_indices = groups[dst_key]
            submat = row_submat[:, dst_indices]
            
            if submat.nnz > 0:
                # --- MEMORY VS SPEED TRADEOFF HERE ---
                rows, cols = submat.shape
                # If matrix is small, densify it for faster numpy multiplication
                if (rows * cols) < DENSE_THRESHOLD:
                    # Convert to dense numpy array
                    final_mat = submat.toarray()
                    dense_count += 1
                else:
                    # Keep as sparse CSR
                    final_mat = submat
                
                neighbor_map[src_key].append((dst_key, final_mat))
                precalc_count += 1

    G_NEIGHBOR_MAP = neighbor_map
    print(f"[INFO] Graph built. {precalc_count} edges. {dense_count} converted to dense for speed.")

    # --- 5. Parallel Execution ---
    start_nodes = [k for k in group_keys if k[1] == MIN_LEVEL]
    print(f"[INFO] Starting Search on {len(start_nodes)} start groups using {n_cores} cores...")

    final_results = []
    
    # We use chunksize=1 to keep memory balanced if some tasks are huge
    with multiprocessing.Pool(processes=n_cores) as pool:
        for res_batch in pool.imap_unordered(worker_process_start_node, start_nodes, chunksize=1):
            if res_batch:
                final_results.extend(res_batch)
                
            sys.stdout.write(f"\r[RUNNING] Motifs found so far: {len(final_results)}")
            sys.stdout.flush()

    print("\n[INFO] Search finished. Aggregating results...")

    # --- 6. Save Results ---
    df_final = pd.DataFrame(final_results)
    
    if not df_final.empty:
        expanded_rows = []
        for r in final_results:
            row_data = {k: v for k,v in r.items() if k != 'path_tuple'}
            path = r['path_tuple']
            for idx, (cls, lev) in enumerate(path):
                row_data[f'cls_{idx}'] = cls
                row_data[f'lev_{idx}'] = lev
            expanded_rows.append(row_data)
        
        df_final = pd.DataFrame(expanded_rows)
        df_final = df_final.sort_values(by=['motif_count'], ascending=False)
        
    nh_str = f"_nh{nh}" if nh else ""
    sl_str = "_noSelfLoop" if no_self_loops else ""
    jump_str = f"_jump{max_jump}"
    
    out_name = f"motifs_nl{'-'.join(map(str, NL_WINDOW))}{nh_str}{jump_str}{sl_str}.csv"
    out_path = os.path.join(out_folder, out_name)
    
    df_final.to_csv(out_path, index=False)
    
    print(f"\n[SUCCESS] Completed in {time.time()-start_time:.2f}s")
    print(f"[OUTPUT] {out_path}")
    
    if not df_final.empty:
        print(f"\n[TOP 10 MOTIFS]")
        pd.set_option('display.max_colwidth', 100)
        print(df_final[['structure_str', 'motif_count']].head(10).to_string(index=False))
    else:
        print(f"[WARN] No motifs found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Dense/Sparse Metagraph Topology")
    parser.add_argument("--window", required=True, help="Level window, E.g.: 1-2-3")
    parser.add_argument("--outdir", default="metagraph_csv")
    parser.add_argument("--nh", type=int, default=None, help="Max distinct classes per level")
    parser.add_argument("--max_jump", type=int, default=1, help="Maximum level distance")
    parser.add_argument("--no_self_loops", action="store_true", help="Prevent ClassA->ClassA")
    parser.add_argument("--cores", type=int, default=max(1, multiprocessing.cpu_count() - 1), 
                        help="Number of CPU cores to use")
    
    args = parser.parse_args()

    try:
        NL_WINDOW = [int(x) for x in args.window.split("-")]
    except ValueError:
        print("Error: Window format must be integers separated by hyphens")
        sys.exit(1)

    out_folder = os.path.join(args.outdir, args.window)
    
    compute_metagraph_topology(
        NL_WINDOW, 
        out_folder, 
        nh=args.nh, 
        max_jump=args.max_jump,
        no_self_loops=args.no_self_loops,
        n_cores=args.cores
    )