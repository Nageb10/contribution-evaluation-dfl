"""
gossip_strategy.py

Implements gossip-based decentralized federated learning.
Replaces the central server aggregation with peer-to-peer
model exchange across a communication graph.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from logging import INFO

from flwr.common import logger

from fltabular.gossip_topology import get_topology
from fltabular.model_logger import GossipModelLogger
from fltabular.task import IncomeClassifier, evaluate, set_weights, get_weights, train


def weighted_average_params(params_list: List[List[np.ndarray]], weights: List[int]) -> List[np.ndarray]:
    total = sum(weights)
    averaged = []
    for layer_idx in range(len(params_list[0])):
        layer_avg = sum(
            w * params_list[i][layer_idx]
            for i, w in enumerate(weights)
        ) / total
        averaged.append(layer_avg)
    return averaged


def run_gossip_simulation(
    num_rounds: int,
    num_clients: int,
    client_ids: List[int],
    train_loaders: List[Any],
    test_loaders: List[Any],
    topology_type: str = "ring",
    edge_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[GossipModelLogger, List[Dict]]:

    # Build communication graph
    topology = get_topology(topology_type, num_clients, edge_prob, seed)
    logger.log(INFO, f"Gossip topology ({topology_type}): {topology}")

    # Initialize one model per client
    models = [IncomeClassifier() for _ in range(num_clients)]

    # Initialize logger
    gossip_logger = GossipModelLogger(num_clients=num_clients, client_ids=client_ids)

    # Track number of examples per client
    num_examples = [len(train_loaders[i].dataset) for i in range(num_clients)]

    round_metrics = []

    for rnd in range(1, num_rounds + 1):
        logger.log(INFO, f"[GOSSIP ROUND {rnd}]")

        # Save current model params before exchange
        current_params = [get_weights(models[i]) for i in range(num_clients)]

        # Each client aggregates models from its neighbors + itself
        new_params = []
        for i in range(num_clients):
            neighbor_indices = topology[i]
            all_indices = [i] + neighbor_indices
            params_to_avg = [current_params[j] for j in all_indices]
            weights_to_avg = [num_examples[j] for j in all_indices]
            aggregated = weighted_average_params(params_to_avg, weights_to_avg)
            new_params.append(aggregated)

        # Each client loads aggregated params then trains locally
        for i in range(num_clients):
            set_weights(models[i], new_params[i])
            train(models[i], train_loaders[i])

        # Evaluate and log each client's model
        rnd_metrics = {"round": rnd, "clients": {}}
        for i in range(num_clients):
            client_id = client_ids[i]
            params_after = get_weights(models[i])

            gossip_logger.log_round(
                round_num=rnd,
                client_id=client_id,
                params=params_after,
                num_examples=num_examples[i]
            )

            loss, acc = evaluate(models[i], test_loaders[i])
            rnd_metrics["clients"][client_id] = {"loss": loss, "accuracy": acc}
            logger.log(INFO, f"  Client {client_id}: loss={loss:.4f}, acc={acc:.4f}")

        round_metrics.append(rnd_metrics)

    logger.log(INFO, f"Gossip simulation complete. "
                     f"{gossip_logger.num_rounds_logged()} rounds logged.")

    return gossip_logger, round_metrics