#!/bin/bash
ROUNDS=50
REPS=10
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

TOPOLOGIES=("ring" "full" "erdos_renyi")
PARTITIONS=("iid" "dirichlet")
ALPHAS=("0.1" "1.0")
METHODS=("loo" "shapley")

for TOPO in "${TOPOLOGIES[@]}"; do
  for METHOD in "${METHODS[@]}"; do

    # IID condition
    for REP in $(seq 1 $REPS); do
      echo "Running: topology=$TOPO partition=iid method=$METHOD rep=$REP"
      python3 -m fltabular.run_gossip \
        --topology $TOPO \
        --partition iid \
        --method $METHOD \
        --rounds $ROUNDS \
        --seed $REP \
        >> "$LOG_DIR/${TOPO}_iid_${METHOD}.log" 2>&1
    done

    # Non-IID conditions
    for ALPHA in "${ALPHAS[@]}"; do
      for REP in $(seq 1 $REPS); do
        echo "Running: topology=$TOPO partition=dirichlet alpha=$ALPHA 
method=$METHOD rep=$REP"
        python3 -m fltabular.run_gossip \
          --topology $TOPO \
          --partition dirichlet \
          --alpha $ALPHA \
          --method $METHOD \
          --rounds $ROUNDS \
          --seed $REP \
          >> "$LOG_DIR/${TOPO}_dirichlet${ALPHA}_${METHOD}.log" 2>&1
      done
    done

  done
done

echo "All main experiments complete."
