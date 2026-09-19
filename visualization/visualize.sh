#!/bin/bash

# >>> ACTIVATE CONDA <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

PY=python
SCRIPT="visualize.py"

IN_DIR="../computation/results_csv"
OUT_DIR="results_png"

# PARAMETERS
TOP_N=3  # Number of top motifs to visualize
WINDOWS=(
    "1-2-3"
)

NH_VALUES=(1)
MAX_JUMPS=(1)

K_VALUES=(1 5) 

mkdir -p $OUT_DIR

for W in "${WINDOWS[@]}"; do
    for NH in "${NH_VALUES[@]}"; do
        for NJ in "${MAX_JUMPS[@]}"; do
            for K in "${K_VALUES[@]}"; do # AGGIUNTO: ciclo for per K
                
                echo ">>> nl=$W | nh=$NH | nj=$NJ | k=$K (Top $TOP_N)"
                
                $PY $SCRIPT \
                    --window "$W" \
                    --nh "$NH" \
                    --max_jump "$NJ" \
                    --top_n "$TOP_N" \
                    --k "$K" \
                    --indir "$IN_DIR" \
                    --outdir "$OUT_DIR"
                echo ""
                
            done
        done
    done
done

echo ">>> Results saved in $OUT_DIR"