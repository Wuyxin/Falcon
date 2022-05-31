#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python -m run.main_pyg --cfg run/configs/pyg/example_node.yaml --repeat 3 # node classification
python -m run.main_pyg --cfg run/configs/pyg/example_link.yaml --repeat 3 # link prediction
python -m run.main_pyg --cfg run/configs/pyg/example_graph.yaml --repeat 3 # graph classification
