#!/bin/bash

# >>> ACTIVATE CONDA <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

PY=python
SCRIPT="visualize.py"

IN_DIR="../computation/results_csv"
OUT_DIR="results_png"

# PARAMETERS
TOP_N=10  # Number of top motifs to visualize
WINDOWS=(
    "1-2-3-4-5"
)

# PARAMETERS
NH_VALUES=(1)
MAX_JUMPS=(4)

mkdir -p $OUT_DIR

for W in "${WINDOWS[@]}"; do
    for NH in "${NH_VALUES[@]}"; do
        for NJ in "${MAX_JUMPS[@]}"; do
            echo ">>> nl=$W | nh=$NH | nj=$NJ (Top $TOP_N)"
            
            $PY $SCRIPT \
                --window "$W" \
                --nh "$NH" \
                --max_jump "$NJ" \
                --top_n "$TOP_N" \
                --indir "$IN_DIR" \
                --outdir "$OUT_DIR"
            echo ""
        done
    done
done

echo ">>> Results saved in $OUT_DIR"