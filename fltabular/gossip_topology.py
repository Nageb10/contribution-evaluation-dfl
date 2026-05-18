"""
gossip_topology.py

Defines the communication graph for gossip-based decentralized FL. 
Support three topology types:

    - 'ring': each node connects to its two immediate neighbors (sparse)
    - 'full': every node connects to every other node (dense)
    - 'erdos_renyi': random graph with a given edge probability (medium)
"""

import random
from typing import Dict, List

def build_ring_toplogy(num_clients: int) -> Dict[int, List[int]]:
    neighbors = {}
    for i in range (num_clients):
        left = (i - 1) % num_clients
        right = (i + 1) % num_clients
        neighbors[i] = [left, right]
    return neighbors


def build_full_topology(num_clients: int) -> Dict[int, List[int]]:
    neighbors = {}
    for i in range(num_clients):
        neighbors[i] = [j for j in range(num_clients) if j != i]
    return neighbors


def build_erdos_renyi_topology(num_clients: int, edge_prob: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
    
    random.seed(seed)
    neighbors = {i: [] for i in range(num_clients)}

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            if random.random() < edge_prob:
                neighbors[i].append(j)
                neighbors[j].append(i)

    for i in range(num_clients):
        if len(neighbors[i]) == 0:
            right = (i + 1) % num_clients
            neighbors[i].append(right)
            neighbors[right].append(i)

    return neighbors

def get_topology(topology_type: str, num_clients: int, edge_prob: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
    if topology_type == "ring":
        return build_ring_toplogy(num_clients)
    elif topology_type == "full":
        return build_full_topology(num_clients)
    elif topology_type == "erdos_renyi":
        return build_erdos_renyi_topology(num_clients, edge_prob, seed)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}. " f"Choose from 'ring, 'full, 'erdos_renyi")
