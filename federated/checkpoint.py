"""
federated/checkpoint.py
-----------------------
Saves and restores federated experiment state for crash recovery.

Resume Rule:
- The server continues training from ckpt["round"] + 1.
- Checkpoints are saved atomically only after a round is completely finished
  and logged, preventing duplicate rounds.
- When resuming, log files should skip or truncate entries where round > ckpt["round"].

Design Note:
- Optimizer state is intentionally not stored here: local optimizers are reconfigured
  freshly each federated round, so only the aggregated global model weights matter.
"""

#Block 1: Imports
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)



#Block 2: Atomic save
def atomic_save(obj: Any, path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target)
    return target



#Block 3: Save checkpoint
def save_checkpoint(
    ckpt_dir: Union[str, Path],
    round_num: int,
    global_state: Dict[str, torch.Tensor],
    selector_state: Any,
    tracker_state: Any,
    counters: Dict[str, Any],
    config: Dict[str, Any],
    wandb_run_id: Optional[str] = None,
    save_every_n: int = 10,
) -> Path:
    out_dir = Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    cpu_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()}

    ckpt = {
        "round": round_num,
        "global_state": cpu_state,
        "selector_state": selector_state,
        "tracker_state": tracker_state,
        "counters": counters,
        "rng": rng_state,
        "config": config,
        "wandb_run_id": wandb_run_id,
    }

    latest_path = out_dir / "latest.pt"
    atomic_save(ckpt, latest_path)

    if save_every_n > 0 and round_num % save_every_n == 0:
        periodic_path = out_dir / f"round_{round_num:03d}.pt"
        atomic_save(ckpt, periodic_path)

    return latest_path



#Block 4: Load checkpoint & restore RNG
def load_checkpoint(path: Optional[Union[str, Path]]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def restore_rng(ckpt: dict):
    rng = ckpt.get("rng", {})
    if "random" in rng:
        random.setstate(rng["random"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:
            logger.warning("Could not restore CUDA RNG state (possible GPU device count mismatch): %s", e)



#Block 5: Self-test
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Setup sample state and RNG seed
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        fake_state = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        counters = {"participation": {"A": 3, "B": 2}, "consecutive_skips": {"C": 1}}
        config = {"seed": 42, "rounds": 50}

        # Snapshot next values before checkpoint save
        next_py_rand = random.random()
        next_np_rand = np.random.rand()
        next_th_rand = torch.rand(1).item()

        # Reset back so save_checkpoint captures that point
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # 2. Save checkpoint (round 10 -> writes latest.pt and round_010.pt)
        ckpt_file = save_checkpoint(
            ckpt_dir=tmp_path,
            round_num=10,
            global_state=fake_state,
            selector_state={"strategy": "random"},
            tracker_state={"total_bytes": 100},
            counters=counters,
            config=config,
            wandb_run_id="run_123",
            save_every_n=10,
        )

        assert (tmp_path / "latest.pt").exists()
        assert (tmp_path / "round_010.pt").exists()

        # Advance RNG forward arbitrarily
        _ = random.random(), np.random.rand(), torch.rand(1)

        # 3. Simulate leftover .tmp file
        (tmp_path / "latest.pt.tmp").write_text("dummy interrupted write")

        # 4. Load checkpoint and verify values
        loaded = load_checkpoint(tmp_path / "latest.pt")
        assert loaded is not None
        assert loaded["round"] == 10
        assert torch.equal(loaded["global_state"]["weight"], fake_state["weight"])
        assert loaded["counters"] == counters
        assert loaded["config"] == config
        assert loaded["wandb_run_id"] == "run_123"

        # 5. Restore RNG and verify repeatability
        restore_rng(loaded)
        assert random.random() == next_py_rand
        assert np.random.rand() == next_np_rand
        assert torch.rand(1).item() == next_th_rand

        # 6. Non-existent path returns None
        assert load_checkpoint(tmp_path / "does_not_exist.pt") is None

