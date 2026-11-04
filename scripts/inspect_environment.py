import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# make "configs" importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config as C

print(f"Environment   : {C.ENV}")
print(f"Project root  : {C.PROJECT_ROOT}")
print(f"BraTS data    : {C.DATA_DIR}")
print(f"Output dir    : {C.OUTPUT_DIR}")
print(f"nnUNet_raw    : {C.NNUNET_RAW}")
print(f"nnUNet_preproc: {C.NNUNET_PREPROCESSED}")
print(f"nnUNet_results: {C.NNUNET_RESULTS}")

print(f"\nPython        : {sys.version.split()[0]}")
try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU           : {p.name}, {p.total_memory / 1024**3:.1f} GB")
    else:
        print("GPU           : none")
except ImportError:
    print("PyTorch       : not installed")

try:
    print(f"nnunetv2      : {version('nnunetv2')}")
except PackageNotFoundError:
    print("nnunetv2      : not installed")

C.require_data_dir()  # raises a clear error (non-zero exit) if the data path is wrong
print("\nData dir OK")