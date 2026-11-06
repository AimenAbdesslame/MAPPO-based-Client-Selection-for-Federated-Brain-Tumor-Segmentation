import os
from pathlib import Path

import yaml

# ==========================================
# 1. PROJECT ROOT & ENVIRONMENT
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ENV = "kaggle" if IS_KAGGLE else "local"

with open(PROJECT_ROOT / "configs" / "base.yaml") as f:
    CFG = yaml.safe_load(f)


# ==========================================
# 2. PATHS  (env var > environments.<ENV> > paths)
# ==========================================
def _path(env_var, key):
    val = (
        (os.environ.get(env_var) if env_var else None)
        or CFG["environments"][ENV].get(key)
        or CFG["paths"].get(key)
    )
    if val is None:
        return None
    p = Path(os.path.expandvars(val)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_DIR = _path("BRATS_DATA_DIR", "brats_data_dir")  # never guessed, never created
OUTPUT_DIR = _path("PROJECT_OUTPUT_DIR", "output_dir")
NNUNET_RAW = _path("nnUNet_raw", "nnunet_raw")
NNUNET_PREPROCESSED = _path("nnUNet_preprocessed", "nnunet_preprocessed")
NNUNET_RESULTS = _path("nnUNet_results", "nnunet_results")
HOLDOUT_DIR = _path(None, "holdout_dir")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
INIT_DIR = OUTPUT_DIR / "initialization"
LOG_DIR = OUTPUT_DIR / "logs"
PARTITION_DIR = OUTPUT_DIR / "partitions"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in (CHECKPOINT_DIR, FIGURES_DIR, INIT_DIR, LOG_DIR, PARTITION_DIR, RESULTS_DIR,
          NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, HOLDOUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# nnU-Net reads these from os.environ
os.environ["nnUNet_raw"] = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


def require_data_dir():
    """Call this in scripts that read BraTS. Fails clearly if the path is missing."""
    if DATA_DIR is None or not DATA_DIR.exists():
        raise FileNotFoundError(
            f"BraTS data dir not found: {DATA_DIR}\n"
            "Set BRATS_DATA_DIR, e.g.\n"
            "  PowerShell: $env:BRATS_DATA_DIR='D:\\path'\n"
            "  Linux:      export BRATS_DATA_DIR=/path\n"
            "  Kaggle:     os.environ['BRATS_DATA_DIR'] = '/kaggle/input/...'"
        )
    return DATA_DIR


# ==========================================
# 3. DATASET
# ==========================================
DATASET_ID = CFG["dataset"]["dataset_id"]
DATASET_NAME = CFG["dataset"]["dataset_name"]
MODALITIES = CFG["dataset"]["modalities"]  # {"FLAIR": 0, "T1": 1, ...}


# ==========================================
# 4. SPLIT & CLIENTS
# ==========================================
SEED = CFG["split"]["seed"]
SPLIT_COUNTS = (CFG["split"]["train"], CFG["split"]["val"], CFG["split"]["test"])
N_TRAIN, N_VAL, N_TEST = SPLIT_COUNTS

CLIENT_NAMES = CFG["clients"]["names"]
CLIENT_PROPORTIONS = CFG["clients"]["proportions"]
CLIENT_COSTS = CFG["clients"]["cost_multipliers"]

# The only checks worth keeping: a typo here silently ruins every experiment
assert abs(sum(CLIENT_PROPORTIONS) - 1.0) < 1e-6, "client proportions must sum to 1"
assert len(CLIENT_NAMES) == len(CLIENT_PROPORTIONS) == len(CLIENT_COSTS)
for _i, _row in enumerate(CFG["clients"]["quartile_mix"]):
    assert abs(sum(_row) - 1.0) < 1e-6, f"quartile_mix row {_i} must sum to 1"