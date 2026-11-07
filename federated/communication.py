"""
federated/communication.py
--------------------------
Calculates and tracks communication volume across federated learning rounds.

Included:
- Model weights download (server -> selected clients)
- Model weights upload (selected clients -> server)

Excluded:
- Lightweight control signals and coordination messages
- Partition/split metadata exchanges
- Validation metric reports and logs
"""

#Block 1: Imports and model size
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from configs.config import CLIENT_COSTS
except Exception:
    CLIENT_COSTS = [1.0, 1.5, 2.0, 3.0]


def model_bytes(state: dict) -> int:
    return sum(v.numel() * v.element_size() for v in state.values())



#Block 2: Round bytes and simulated round cost
def round_bytes(m_bytes: int, n_selected: int) -> int:
    return 2 * m_bytes * n_selected


def round_cost(m_bytes: int, selected_idx: List[int], costs: List[float] = CLIENT_COSTS) -> float:
    return 2.0 * m_bytes * sum(costs[i] for i in selected_idx)



#Block 3: Communication tracker
class CommunicationTracker:
    def __init__(self, m_bytes: int, costs: Optional[List[float]] = None):
        self.m_bytes = m_bytes
        self.costs = costs if costs is not None else CLIENT_COSTS
        self.total_bytes: int = 0
        self.total_cost: float = 0.0

    def add(self, selected_idx: List[int]):
        self.total_bytes += round_bytes(self.m_bytes, len(selected_idx))
        self.total_cost += round_cost(self.m_bytes, selected_idx, self.costs)

    def state(self) -> dict:
        return {
            "m_bytes": self.m_bytes,
            "costs": self.costs,
            "total_bytes": self.total_bytes,
            "total_cost": self.total_cost,
        }

    def load(self, state: dict):
        self.m_bytes = state["m_bytes"]
        self.costs = state["costs"]
        self.total_bytes = state["total_bytes"]
        self.total_cost = state["total_cost"]



#Block 4: Self-test
if __name__ == "__main__":
    fake_state = {"w": torch.zeros(10, dtype=torch.float32)}
    b = model_bytes(fake_state)
    assert b == 40, f"Expected 40 bytes, got {b}"

    rb = round_bytes(b, 2)
    assert rb == 160, f"Expected 160 bytes, got {rb}"

    costs = [1.0, 1.5, 2.0, 3.0]
    rc = round_cost(b, [0, 3], costs)
    assert rc == 320.0, f"Expected 320.0 cost, got {rc}"

    tracker = CommunicationTracker(b, costs)
    tracker.add([0, 3])
    assert tracker.total_bytes == 160
    assert tracker.total_cost == 320.0

    saved = tracker.state()
    tracker_restored = CommunicationTracker(0)
    tracker_restored.load(saved)
    assert tracker_restored.total_bytes == 160
    assert tracker_restored.total_cost == 320.0

