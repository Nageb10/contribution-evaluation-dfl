"""
analyze_results.py

Reads all 180 gossip experiment JSON files and computes:
- Score stability (std across seeds per condition)
- Discriminatory power (score spread under non-IID)
- Summary statistics per condition

For score agreement (Spearman/Kendall vs CFL baseline),
CFL results need to be added separately once available.

Usage:
    python3 analyze_results.py
"""

import json
import os
import numpy as np
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict

RESULTS_DIR = "gossip_results"
CLIENT_IDS = ["1", "2", "3", "4", "5"]


def load_all_results(results_dir):
    """Load all JSON result files into a nested dict."""
    results = defaultdict(list)
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        config = data["config"]
        key = (
            config["topology"],
            config["partition_type"],
            str(data["config"].get("dirichlet_alpha",
                config.get("num_clients", ""))),
            config["contrib_method"]
        )
        # Extract CE scores as ordered list
        scores = [data["ce_scores"][cid] for cid in CLIENT_IDS]
        results[fname] = data
    return results


def group_by_condition(results):
    """Group results by (topology, partition, alpha, method)."""
    groups = defaultdict(list)
    for fname, data in results.items():
        config = data["config"]
        topo = config["topology"]
        partition = config["partition_type"]
        method = config["contrib_method"]

        # Extract alpha from filename
        if "alpha0.1" in fname:
            alpha = "0.1"
        elif "alpha1.0" in fname:
            alpha = "1.0"
        else:
            alpha = "iid"

        key = (topo, partition, alpha, method)
        scores = [data["ce_scores"][cid] for cid in CLIENT_IDS]
        groups[key].append(scores)
    return groups


def compute_stability(groups):
    """
    For each condition, compute mean std of CE scores across seeds.
    Lower = more stable.
    """
    stability = {}
    for key, score_list in groups.items():
        arr = np.array(score_list)  # shape: (num_seeds, num_clients)
        per_client_std = arr.std(axis=0)
        stability[key] = {
            "mean_std": per_client_std.mean(),
            "per_client_std": per_client_std.tolist(),
            "mean_scores": arr.mean(axis=0).tolist()
        }
    return stability


def compute_discriminatory_power(groups):
    """
    For each condition, compute the score spread across clients.
    Higher spread = better discriminatory power.
    """
    power = {}
    for key, score_list in groups.items():
        arr = np.array(score_list)  # shape: (num_seeds, num_clients)
        mean_scores = arr.mean(axis=0)
        score_range = mean_scores.max() - mean_scores.min()
        score_std = mean_scores.std()
        power[key] = {
            "score_range": score_range,
            "score_std": score_std,
            "mean_scores": mean_scores.tolist(),
            "client_ranking": np.argsort(-mean_scores).tolist()
        }
    return power

def analyze_noisy_detection(results):
    """
    For noisy client experiments, check whether client 5
    consistently receives the lowest CE score across seeds.
    """
    detection = defaultdict(list)

    for fname, data in results.items():
        if "noisy5" not in fname:
            continue

        config = data["config"]
        topo = config["topology"]
        partition = config["partition_type"]
        method = config["contrib_method"]

        if "alpha0.1" in fname:
            alpha = "0.1"
        elif "alpha1.0" in fname:
            alpha = "1.0"
        else:
            alpha = "iid"

        key = (topo, partition, alpha, method)
        scores = {cid: data["ce_scores"][cid] for cid in CLIENT_IDS}

        # Check if client 5 has the lowest score
        min_client = min(scores, key=scores.get)
        detected = (min_client == "5")
        detection[key].append(detected)

    # Compute detection rate per condition
    detection_rate = {}
    for key, detections in detection.items():
        detection_rate[key] = {
            "detection_rate": sum(detections) / len(detections),
            "detected_count": sum(detections),
            "total_runs": len(detections)
        }
    return detection_rate


def print_noisy_detection_table(detection_rate):
    print("\n" + "="*70)
    print("NOISY CLIENT DETECTION RATE (client 5 ranked lowest)")
    print("="*70)
    print(f"{'Topology':<15} {'Partition':<12} {'Alpha':<8} "
          f"{'Method':<12} {'Detection Rate':>16}")
    print("-"*70)
    for key in sorted(detection_rate.keys()):
        topo, partition, alpha, method = key
        rate = detection_rate[key]["detection_rate"]
        count = detection_rate[key]["detected_count"]
        total = detection_rate[key]["total_runs"]
        print(f"{topo:<15} {partition:<12} {alpha:<8} {method:<12} "
              f"{rate:>14.1%} ({count}/{total})")


def compute_rank_agreement(groups):
    """
    Compute Spearman and Kendall rank correlations between
    topologies within the same data condition and method.
    This shows how much topology affects CE rankings.
    """
    correlations = {}
    methods = ["loo", "shapley"]
    conditions = [("iid", "iid"), ("dirichlet", "0.1"), ("dirichlet", "1.0")]
    topologies = ["ring", "full", "erdos_renyi"]

    for method in methods:
        for partition, alpha in conditions:
            # Get mean scores for each topology
            topo_scores = {}
            for topo in topologies:
                key = (topo, partition, alpha, method)
                if key in groups:
                    arr = np.array(groups[key])
                    topo_scores[topo] = arr.mean(axis=0)

            # Compare each pair of topologies
            topo_list = list(topo_scores.keys())
            for i in range(len(topo_list)):
                for j in range(i + 1, len(topo_list)):
                    t1, t2 = topo_list[i], topo_list[j]
                    s1, s2 = topo_scores[t1], topo_scores[t2]
                    rho, p_rho = spearmanr(s1, s2)
                    tau, p_tau = kendalltau(s1, s2)
                    corr_key = (method, partition, alpha, t1, t2)
                    correlations[corr_key] = {
                        "spearman_rho": round(rho, 4),
                        "spearman_p": round(p_rho, 4),
                        "kendall_tau": round(tau, 4),
                        "kendall_p": round(p_tau, 4)
                    }
    return correlations


def print_stability_table(stability):
    print("\n" + "="*70)
    print("SCORE STABILITY (mean std across 10 seeds)")
    print("="*70)
    print(f"{'Topology':<15} {'Partition':<12} {'Alpha':<8} "
          f"{'Method':<10} {'Mean Std':>10}")
    print("-"*70)
    for key in sorted(stability.keys()):
        topo, partition, alpha, method = key
        val = stability[key]["mean_std"]
        print(f"{topo:<15} {partition:<12} {alpha:<8} {method:<10} {val:>10.6f}")


def print_discriminatory_power_table(power):
    print("\n" + "="*70)
    print("DISCRIMINATORY POWER (score range across clients)")
    print("="*70)
    print(f"{'Topology':<15} {'Partition':<12} {'Alpha':<8} "
          f"{'Method':<10} {'Range':>10} {'Std':>10}")
    print("-"*70)
    for key in sorted(power.keys()):
        topo, partition, alpha, method = key
        r = power[key]["score_range"]
        s = power[key]["score_std"]
        print(f"{topo:<15} {partition:<12} {alpha:<8} {method:<10} "
              f"{r:>10.6f} {s:>10.6f}")


def print_rank_correlations(correlations):
    print("\n" + "="*70)
    print("RANK CORRELATIONS BETWEEN TOPOLOGIES")
    print("="*70)
    print(f"{'Method':<10} {'Partition':<12} {'Alpha':<6} "
          f"{'Topo 1':<14} {'Topo 2':<14} {'Spearman ρ':>12} {'Kendall τ':>12}")
    print("-"*70)
    for key in sorted(correlations.keys()):
        method, partition, alpha, t1, t2 = key
        val = correlations[key]
        print(f"{method:<10} {partition:<12} {alpha:<6} {t1:<14} {t2:<14} "
              f"{val['spearman_rho']:>12.4f} {val['kendall_tau']:>12.4f}")


def save_summary(stability, power, correlations, detection_rate, output_file="analysis_summary.json"):
    summary = {
        "stability": {str(k): v for k, v in stability.items()},
        "discriminatory_power": {str(k): v for k, v in power.items()},
        "rank_correlations": {str(k): v for k, v in correlations.items()},
        "noisy_detection": {str(k): v for k, v in detection_rate.items()}
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {output_file}")


if __name__ == "__main__":
    print(f"Loading results from {RESULTS_DIR}...")
    results = load_all_results(RESULTS_DIR)
    print(f"Loaded {len(results)} result files.")

    groups = group_by_condition(results)
    print(f"Grouped into {len(groups)} experimental conditions.")

    stability = compute_stability(groups)
    power = compute_discriminatory_power(groups)
    correlations = compute_rank_agreement(groups)
    detection_rate = analyze_noisy_detection(results)      

    print_stability_table(stability)
    print_discriminatory_power_table(power)
    print_rank_correlations(correlations)
    print_noisy_detection_table(detection_rate)            

    save_summary(stability, power, correlations, detection_rate)