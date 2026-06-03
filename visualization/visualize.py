#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import argparse
import sys

# --- Color Map ---
CLASS_COLOR_MAP = {
    'optic': '#1f77b4',
    'central': '#2ca02c',
    'visual_centrifugal': '#9467bd',
    'visual_projection': '#ff7f0e',
    'ascending': '#d62728',
    'sensory': '#17becf',
    'descending': '#e377c2',
    'motor': '#8c564b',
    'DEFAULT': '#7f7f7f'
}

def visualize_motifs(WINDOW_STR, NH_VALUE, MAX_JUMP, TOP_N, BASE_IN_DIR, BASE_OUT_DIR):
    """
    Load motif results and visualize Top N.
    """
    # --- 1. File Search ---
    INPUT_DIR = os.path.join(BASE_IN_DIR, WINDOW_STR)
    found_file = None
    base_name = f"motifs_nl{WINDOW_STR}"
    
    if os.path.exists(INPUT_DIR):
        candidates = [f for f in os.listdir(INPUT_DIR) if f.startswith(base_name) and f.endswith(".csv")]
        
        if NH_VALUE:
            candidates = [f for f in candidates if f"_nh{NH_VALUE}" in f]
            
        if MAX_JUMP:
            candidates = [f for f in candidates if f"_jump{MAX_JUMP}" in f]
        
        if candidates:
            candidates.sort(reverse=True)
            found_file = os.path.join(INPUT_DIR, candidates[0])
    
    if not found_file:
        print(f"[ERROR] No compatible CSV file found in {INPUT_DIR}")
        return
        
    OUTPUT_DIR = os.path.join(BASE_OUT_DIR, WINDOW_STR)
    subfolder_name = f"nh{NH_VALUE if NH_VALUE else 'All'}_nj{MAX_JUMP if MAX_JUMP else '1'}"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, subfolder_name)

    # --- 2. Load Data ---
    try:
        df_motifs = pd.read_csv(found_file)
    except Exception:
        return

    if df_motifs.empty: return

    df_motifs = df_motifs.sort_values(by='motif_count', ascending=False)
    df_top = df_motifs.head(TOP_N).copy()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 3. Visualization Loop ---
    for idx, row in df_top.iterrows():
        rank = idx + 1
        count = row['motif_count']
        chain_len = row['chain_length']
        
        # --- 3.1 Node Construction ---
        node_ids = []
        node_props = []
        seen_nodes = {} 
        
        for k in range(chain_len):
            cls = row[f'cls_{k}']
            lev = row[f'lev_{k}']
            curr_prop = (cls, lev)
            
            if curr_prop in seen_nodes:
                current_id = seen_nodes[curr_prop]
            else:
                current_id = f"{cls}_L{lev}_{k}"
                seen_nodes[curr_prop] = current_id
            
            node_ids.append(current_id)
            node_props.append(curr_prop)

        # --- 3.2 Graph ---
        G = nx.DiGraph()
        
        first_appearance = {}
        for i, nid in enumerate(node_ids):
            if nid not in first_appearance:
                first_appearance[nid] = i

        unique_nodes_map = {v: k for k, v in seen_nodes.items()}
        for nid, (cls, lev) in unique_nodes_map.items():
            G.add_node(nid, node_class=cls, level=lev, label_idx=first_appearance[nid])
        
        for i in range(len(node_ids) - 1):
            u = node_ids[i]
            v = node_ids[i+1]
            if not G.has_edge(u, v):
                G.add_edge(u, v)

        path_str = "_".join([f"{p[0][:3]}L{p[1]}" for p in node_props])
        if len(path_str) > 60: path_str = path_str[:60] + "..."
        out_filename = f"Rank{rank}_Count{count}_{path_str}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # --- 3.3 Layout ---
        ALL_LEVELS = range(1, 6)
        nodes_by_level = {l: [] for l in ALL_LEVELS}
        for node in G.nodes():
            lev = G.nodes[node]['level']
            if lev in nodes_by_level: nodes_by_level[lev].append(node)
        
        for lev in nodes_by_level:
            nodes_by_level[lev].sort(key=lambda n: G.nodes[n]['label_idx'])

        pos = {}
        spacing_x = 4.0
        spacing_y = 4.5
        
        for level in ALL_LEVELS:
            nodes = nodes_by_level[level]
            if not nodes: continue
            n_nodes = len(nodes)
            start_x = -((n_nodes - 1) * spacing_x) / 2
            for i, node_id in enumerate(nodes):
                pos[node_id] = (start_x + i * spacing_x, -level * spacing_y)

        # --- 3.4 Drawing ---
        fig, ax = plt.subplots(figsize=(12, 16))

        node_colors = [CLASS_COLOR_MAP.get(G.nodes[n]['node_class'], '#7f7f7f') for n in G.nodes()]
        labels = {n: G.nodes[n]['node_class'].replace('visual_', 'v_')[:12] for n in G.nodes()}

        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2500, 
                               alpha=0.9, edgecolors='black', linewidths=0)
        
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=15, 
                                font_weight='normal', font_family='sans-serif')

        # Draw Edges
        node_radius = np.sqrt(2500) / 100.0
        
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            if u == v: # Self-loop
                concentric_radius = node_radius + 0.15 
                circle = Circle((x1, y1), concentric_radius, fill=False, 
                                edgecolor='#1f77b4', linewidth=3.5, alpha=0.8, zorder=0)
                ax.add_patch(circle)
            else: # Normal Edge
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0:
                    start_x = x1 + (dx/dist) * node_radius
                    start_y = y1 + (dy/dist) * node_radius
                    end_x = x2 - (dx/dist) * (node_radius * 1.0)
                    end_y = y2 - (dy/dist) * (node_radius * 1.0)
                    
                    src_lev = G.nodes[u]['level']
                    tgt_lev = G.nodes[v]['level']
                    
                    if tgt_lev < src_lev: color = 'red' 
                    elif tgt_lev == src_lev: color = '#1f77b4' 
                    else: color = '#666666'
                    
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=color,
                        connectionstyle='arc3,rad=0.18', 
                        zorder=5
                    )
                    ax.add_patch(arrow)

        # Grid
        spacing_y_grid = 4.5
        for level in list(ALL_LEVELS)[:-1]:
             y_h = -level * spacing_y_grid - (spacing_y_grid/2)
             ax.axhline(y=y_h, color='lightgray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

        # Dynamic Label Placement
        if pos:
            all_x_coords = [p[0] for p in pos.values()]
            min_x = min(all_x_coords)
        else:
            min_x = -5.0
            
        label_x_pos = min_x - 2.5 

        for level in ALL_LEVELS:
            ax.text(label_x_pos, -level * spacing_y_grid, f"L{level}", 
                    fontsize=16, fontweight='bold', va='center', color='#444')

        # --- Title ---
        nl_val = WINDOW_STR.replace("-", ",")
        nh_val = NH_VALUE if NH_VALUE else "All"
        nj_val = MAX_JUMP if MAX_JUMP else "1"
        
        title_text = (f"Rank #{rank} | Count: {count:,}\n"
                      f"nl: [{nl_val}] | nh: {nh_val} | nj: {nj_val}")
        
        plt.title(title_text, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_ylim(-5 * 4.5 - 2, -2)
        
        limit_x = max(6, abs(label_x_pos) + 1)
        ax.set_xlim(-limit_x, limit_x)
        
        ax.set_aspect('equal') 
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, facecolor='white')
        plt.close(fig)

    print(f"[OK] Completed. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, help="E.g.: 1-2-3")
    parser.add_argument("--nh", type=int, default=None)
    parser.add_argument("--max_jump", type=int, default=None)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--indir", default="metagraph_csv")
    parser.add_argument("--outdir", default="metagraph_png")

    args = parser.parse_args()

    visualize_motifs(
        WINDOW_STR=args.window, 
        NH_VALUE=args.nh, 
        MAX_JUMP=args.max_jump,
        TOP_N=args.top_n, 
        BASE_IN_DIR=args.indir, 
        BASE_OUT_DIR=args.outdir
    )