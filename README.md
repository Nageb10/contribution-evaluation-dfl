# When the Server Is Removed
## Contribution Evaluation Reliability in Decentralized Federated Learning

Code for the master's thesis at Oslo Metropolitan University, 2026.

## Prerequisites
This repository contains only the gossip-based DFL extension files.
To reproduce the full experiments, you first need to clone the 
baseline codebase from Pejo et al.:
https://github.com/m9framar/conteval-pois-Flower

Then copy the files from this repository into the fl-tabular 
directory of that codebase.

## Files in this repository
- `fltabular/gossip_topology.py` — communication graph generation
- `fltabular/model_logger.py` — passive post-hoc model state logger
- `fltabular/gossip_strategy.py` — gossip training loop
- `fltabular/run_gossip.py` — experiment entry point
- `analyze_results.py` — analysis script
- `run_experiments_noisy.sh` — noisy client experiment grid

## Experiment Scripts
- `run_experiments_main.sh` — runs the main experiment grid (180 runs)
- `run_experiments_noisy.sh` — runs the noisy client experiment grid (180 runs)

## Requirements
Python 3.13, Flower 1.27.0, PyTorch 2.11.0

## Usage
python3 -m fltabular.run_gossip --topology ring --partition iid \
--method loo --rounds 50 --seed 1
