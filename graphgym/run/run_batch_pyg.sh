#!/usr/bin/env bash

CONFIG=example_node
GRID=example
REPEAT=3
MAX_JOBS=4
SLEEP=1
MAIN=main_pyg

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python -m run.configs_gen --config run/configs/pyg/${CONFIG}.yaml \
  --grid run/grids/pyg/${GRID}.txt \
  --out_dir run/configs 

#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash run/parallel.sh run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash run/parallel.sh run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash run/parallel.sh run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python -m run.agg_batch --dir results/${CONFIG}_grid_${GRID}
