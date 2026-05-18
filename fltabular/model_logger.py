import numpy as np
from typing import Dict, List, Optional

class GossipModelLogger:
    """
    Logs model parameters and metadata for each client after every gossip round. 
    """

    def __init__(self, num_clients: int, client_ids: List[int]):
        """
        Args:
            num_clients: total number of clients
            client_ids: list of client IDs (e.g. [1, 2, 3, 4, 5])
        """

        self.num_clients = num_clients
        self.client_ids = client_ids

        self.round_logs: Dict[int, Dict[int, Dict]] = {}

    def log_round(self, round_num: int, client_id: int, params: List[np.ndarray], num_examples: int) -> None:
        """
            Log a single client's model state after a gossip round.

            Args:
                round_nm: current gossip round (1-indexed)
                client_id: the client's ID
                params: list of numpy arrays (model weights)
                num_examples: number of training examples this client has
        """

        if round_num not in self.round_logs:
            self.round_logs[round_num] = {}

        self.round_logs[round_num][client_id] = {
            "params": [p.copy() for p in params],
            "num_examples": num_examples
        }

    def get_round(self, round_num: int) -> Optional[Dict[int, Dict]]:
        """
        Retrieve all client states for a given round.
        """

        return self.round_logs.get(round_num, None)
    
    def get_all_rounds(self) -> Dict[int, Dict[int, Dict]]:
        "Return the full log across all rounds."
        return self.round_logs
    
    def get_final_round(self) -> Optional[Dict[int, Dict]]:
        "Return the last logged round's client states"
        if not self.round_logs:
            return None
        last_round = max(self.round_logs.keys())
        return self.round_logs[last_round]
    
    def num_rounds_logged(self) -> int:
        "Return how many rounds have been logged."
        return len (self.round_logs)
    
    def is_round_complete(self, round_num: int) -> bool:
        "Check whether all clients have been logged for a given round."
        if round_num not in self.round_logs:
            return False
        return len(self.round_logs[round_num]) == self.num_clients
