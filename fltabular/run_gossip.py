"""
run_gossip.py

Entry point for running the gossip-based DFL simulation
with post-hoc contribution evaluation.

Usage:
    python3 -m fltabular.run_gossip
"""

import json
import os
from logging import INFO

from flwr.common import logger

from fltabular.task import IncomeClassifier, evaluate, load_data
from fltabular.gossip_strategy import run_gossip_simulation
from fltabular.contribution_strategy import LeaveOneOutStrategy, TorchGTGShapleyStrategy


def run(
    num_clients: int = 5,
    num_rounds: int = 50,
    topology_type: str = "ring",
    partition_type: str = "iid",
    dirichlet_alpha: float = 0.5,
    contrib_method: str = "loo",
    edge_prob: float = 0.5,
    seed: int = 42,
    output_dir: str = "gossip_results",
    noisy_client: int = -1,
):
    # Load data for each client
    logger.log(INFO, f"Loading data — partition={partition_type}, "
                     f"topology={topology_type}, method={contrib_method}")

    train_loaders = []
    test_loaders = []
    for i in range(num_clients):
        train_loader, test_loader = load_data(
            partition_id=i,
            num_partitions=num_clients,
            partition_type=partition_type,
            dirichlet_alpha=dirichlet_alpha,
        )
        # If this client is the noisy client, shuffle its labels
        if i == noisy_client:
            logger.log(INFO, f"Client {i+1} is a noisy client — labels will be shuffled")
            from torch.utils.data import DataLoader, TensorDataset
            import torch
            X, y = zip(*[(x, label) for x, label in train_loader.dataset])
            X = torch.stack(X)
            y = torch.stack(y)
            # Shuffle labels randomly
            perm = torch.randperm(len(y))
            y = y[perm]
            noisy_dataset = TensorDataset(X, y)
            train_loader = DataLoader(noisy_dataset, batch_size=32, shuffle=True)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    client_ids = list(range(1, num_clients + 1))

    # Run gossip simulation
    gossip_logger, round_metrics = run_gossip_simulation(
        num_rounds=num_rounds,
        num_clients=num_clients,
        client_ids=client_ids,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        topology_type=topology_type,
        edge_prob=edge_prob,
        seed=seed,
    )

    # Post-hoc CE evaluation using final round model states
    final_round = gossip_logger.get_final_round()
    client_params = [final_round[cid]["params"] for cid in client_ids]
    client_examples = [final_round[cid]["num_examples"] for cid in client_ids]

    # Build a shared holdout test loader (use client 0's test set)
    holdout_loader = test_loaders[0]

    # Compute global model as weighted average for reference loss
    from fltabular.gossip_strategy import weighted_average_params
    global_params = weighted_average_params(client_params, client_examples)
    global_model = IncomeClassifier()
    from fltabular.task import set_weights
    set_weights(global_model, global_params)
    global_loss, global_acc = evaluate(global_model, holdout_loader)
    logger.log(INFO, f"Global model (final round): loss={global_loss:.4f}, acc={global_acc:.4f}")

    # Select CE strategy
    if contrib_method == "loo":
        strategy = LeaveOneOutStrategy(IncomeClassifier, evaluate)
    else:
        strategy = TorchGTGShapleyStrategy(IncomeClassifier, evaluate)

    # Evaluate contributions
    client_results, ce_metrics = strategy.evaluate_contribution(
        client_ids=client_ids,
        client_params=client_params,
        client_examples=client_examples,
        global_params=global_params,
        global_loss=global_loss,
        global_acc=global_acc,
        test_loader=holdout_loader,
        round_num=num_rounds,
    )

    # Log CE scores
    logger.log(INFO, "==== Contribution scores (post-hoc) ====")
    for cid, result in client_results.items():
        if contrib_method == "loo":
            score = result["loss_contribution"]
        else:
            score = result["shapley_value"]
        logger.log(INFO, f"  Client {cid}: {score:.6f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "config": {
            "topology": topology_type,
            "partition_type": partition_type,
            "contrib_method": contrib_method,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "noisy_client": noisy_client + 1 if noisy_client >= 0 else None,
        },
        "round_metrics": round_metrics,
        "ce_scores": {
            str(cid): (result["loss_contribution"] if contrib_method == "loo"
                       else result["shapley_value"])
            for cid, result in client_results.items()
        },
        "global_loss": global_loss,
        "global_acc": global_acc,
    }

    noisy_tag = f"_noisy{noisy_client+1}" if noisy_client >= 0 else ""
    filename = f"{output_dir}/{topology_type}_{partition_type}_alpha{dirichlet_alpha}_{contrib_method}_seed{seed}{noisy_tag}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    logger.log(INFO, f"Results saved to {filename}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gossip DFL simulation")
    parser.add_argument("--topology", type=str, default="ring",
                        choices=["ring", "full", "erdos_renyi"])
    parser.add_argument("--partition", type=str, default="iid",
                        choices=["iid", "dirichlet"])
    parser.add_argument("--method", type=str, default="loo",
                        choices=["loo", "shapley"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="gossip_results")
    parser.add_argument("--noisy-client", type=int, default=-1, help="Index (0-based) of client to make noisy. -1 = no noisy client.")
    args = parser.parse_args()

    run(
        num_rounds=args.rounds,
        topology_type=args.topology,
        partition_type=args.partition,
        contrib_method=args.method,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
        noisy_client=args.noisy_client,
    )