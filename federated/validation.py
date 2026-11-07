"""
federated/validation.py
-----------------------
Evaluates the global model on the held-out validation cohort.
"""

#Block 1: Imports
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import HOLDOUT_DIR, PARTITION_DIR
from evaluation.dice import dice

try:
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError:
    SimpleITKIO = None
    nnUNetPredictor = None



#Block 2: Validation cases
def val_cases(n: Optional[int] = None) -> List[Tuple[str, List[str], str]]:
    split_file = Path(PARTITION_DIR) / "patient_split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"patient_split.json not found in {PARTITION_DIR}")

    split_data = json.loads(split_file.read_text())
    val_ids = sorted(split_data.get("val", []))
    if n is not None:
        val_ids = val_ids[:n]

    img_dir = Path(HOLDOUT_DIR) / "imagesVal"
    lbl_dir = Path(HOLDOUT_DIR) / "labelsVal"

    cases = []
    for pid in val_ids:
        img_paths = [str(img_dir / f"{pid}_{c:04d}.nii.gz") for c in range(4)]
        lbl_path = str(lbl_dir / f"{pid}.nii.gz")
        cases.append((pid, img_paths, lbl_path))
    return cases



#Block 3: Make predictor
def make_predictor(trainer) -> "nnUNetPredictor":
    if nnUNetPredictor is None:
        raise ImportError("nnunetv2 is not installed or importable.")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=trainer.device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.plans_manager = trainer.plans_manager
    predictor.configuration_manager = trainer.configuration_manager
    predictor.dataset_json = trainer.dataset_json
    predictor.label_manager = trainer.label_manager
    predictor.network = trainer.network
    predictor.allowed_mirroring_axes = None
    predictor.list_of_parameters = [trainer.network.state_dict()]
    return predictor



#Block 4: Predict patient
def predict_patient(predictor: "nnUNetPredictor", image_paths: List[str]) -> np.ndarray:
    img, props = SimpleITKIO().read_images(image_paths)
    seg = predictor.predict_single_npy_array(img, props)
    return seg.astype(np.uint8)



#Block 5: Validate
def validate(trainer, n_val_patients: Optional[int] = None, predictor: Optional["nnUNetPredictor"] = None) -> Dict:
    trainer.network.eval()
    try:
        if predictor is None:
            predictor = make_predictor(trainer)
        else:
            predictor.network = trainer.network
            predictor.list_of_parameters = [trainer.network.state_dict()]

        cases = val_cases(n_val_patients)
        reader = SimpleITKIO()
        per_patient = {}
        t0 = time.perf_counter()

        for pid, img_paths, lbl_path in cases:
            pred = predict_patient(predictor, img_paths)
            gt, _ = reader.read_seg(lbl_path)
            if gt.ndim == 4 and gt.shape[0] == 1:
                gt = gt[0]
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {pid}: pred {pred.shape} vs gt {gt.shape}. "
                    "Check axis ordering between image and label."
                )
            per_patient[pid] = dice(pred, gt)

        elapsed = time.perf_counter() - t0
        mean_dice = float(np.mean(list(per_patient.values()))) if per_patient else 0.0
        return {
            "mean_dice": mean_dice,
            "per_patient": per_patient,
            "seconds": elapsed,
        }
    finally:
        trainer.network.train()



#Block 6: Quick test on Kaggle
if __name__ == "__main__":
    if nnUNetPredictor is None:
        print("nnunetv2 not installed locally; run Block 6 test on Kaggle.")
    else:
        from models.nnunet_adapter import build_trainer, train_local, get_weights

        print("Building trainer for quick validation test...")
        trainer = build_trainer(batch_size=16)

        w_before = get_weights(trainer)
        res_rand = validate(trainer, n_val_patients=3)
        w_after = get_weights(trainer)
        assert all(torch.equal(w_before[k], w_after[k]) for k in w_before), "Trainer weights mutated during validation"

        print(f"Random weights -> Mean Dice: {res_rand['mean_dice']:.4f}, Time: {res_rand['seconds']:.2f}s")
        for pid, d in res_rand["per_patient"].items():
            print(f"  {pid}: {d:.4f}")

        print("Training 50 steps...")
        train_local(trainer, n_steps=50)
        res_trained = validate(trainer, n_val_patients=3)
        print(f"Trained 50 steps -> Mean Dice: {res_trained['mean_dice']:.4f}, Time: {res_trained['seconds']:.2f}s")
        for pid, d in res_trained["per_patient"].items():
            print(f"  {pid}: {d:.4f}")

