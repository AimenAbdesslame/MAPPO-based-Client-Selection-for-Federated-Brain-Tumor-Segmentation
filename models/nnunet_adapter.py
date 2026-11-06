
#Block 1: imports / paths
import json, os, sys
from pathlib import Path

os.environ["nnUNet_compile"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (DATASET_ID, DATASET_NAME, NNUNET_PREPROCESSED,
                             PARTITION_DIR, CLIENT_NAMES)

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PREP_DIR   = Path(NNUNET_PREPROCESSED) / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
PLANS_FILE = PREP_DIR / "nnUNetPlans.json"
DS_JSON    = PREP_DIR / "dataset.json"



#Block 2: write client sub-splits into preprocessed folder
def write_client_splits():
    partitions = json.loads((Path(PARTITION_DIR) / "client_partitions.json").read_text())
    splits_all = [{"train": partitions[name], "val": partitions[name][:2]}
                  for name in CLIENT_NAMES]
    out = PREP_DIR / "splits_final.json"
    out.write_text(json.dumps(splits_all, indent=2))
    return out



#Block 3: build and initialise the trainer
def build_trainer(batch_size: int) -> nnUNetTrainer:
    plans   = json.loads(PLANS_FILE.read_text())
    ds_json = json.loads(DS_JSON.read_text())

    plans["configurations"]["2d"]["batch_size"] = batch_size
    plans["continue_training"] = False

    trainer = nnUNetTrainer(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=ds_json,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.initialize()
    trainer.set_deep_supervision_enabled(True)
    trainer.dataset_class.unpack_dataset(
        str(PREP_DIR),
        overwrite_existing=False,
        num_processes=max(1, os.cpu_count() // 2),
    )
    return trainer



#Block 4: switch to a different client's data split
def set_client(trainer: nnUNetTrainer, idx: int):
    for dl in (getattr(trainer, "dataloader_train", None),
               getattr(trainer, "dataloader_val",   None)):
        if dl is not None and hasattr(dl, "_finish"):
            dl._finish()

    trainer.fold = idx
    trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()



#Block 5: weight transfer (CPU copies, handles DDP .module wrapper)
def get_weights(trainer: nnUNetTrainer) -> dict:
    net = trainer.network
    if hasattr(net, "module"):
        net = net.module
    return {k: v.cpu().clone() for k, v in net.state_dict().items()}

def set_weights(trainer: nnUNetTrainer, state: dict):
    net = trainer.network
    if hasattr(net, "module"):
        net = net.module
    net.load_state_dict(state)



#Block 6: local training for n_steps, returns mean loss
def train_local(trainer: nnUNetTrainer, n_steps: int) -> float:
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    trainer.network.train()
    total = 0.0
    for _ in range(n_steps):
        total += trainer.train_step(next(trainer.dataloader_train))["loss"]
    return float(total / n_steps)
