#!/bin/bash

# >>> ACTIVATE CONDA <<<
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis

PY=python
SCRIPT="compute.py"

WINDOWS=(
    "1-2-3"
)

# PARAMETERS
NH_VALUES=(1)
MAX_JUMPS=(1)
SELF_LOOP_MODES=("no")
K_VALUES=(1 5) 

for W in "${WINDOWS[@]}"; do
    for NH in "${NH_VALUES[@]}"; do
        for J in "${MAX_JUMPS[@]}"; do
            for S in "${SELF_LOOP_MODES[@]}"; do
                for K in "${K_VALUES[@]}"; do

                    echo ">>> nl=$W | nh=$NH | nj=$J | no_self_loops=$S | k=$K"

                    if [[ "$S" == "no" ]]; then
                        NO_SELF="--no_self_loops"
                    else
                        NO_SELF=""
                    fi

                    $PY $SCRIPT \
                        --window "$W" \
                        --nh "$NH" \
                        --max_jump "$J" \
                        --k "$K" \
                        $NO_SELF \
                        --outdir "results_csv"
                    echo ""
                done
            done
        done
    done
done