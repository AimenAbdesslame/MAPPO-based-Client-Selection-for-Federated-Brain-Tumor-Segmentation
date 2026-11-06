#Block 1: Imports and constants
import argparse, json, shutil, sys, tarfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (DATASET_ID, DATASET_NAME, MODALITIES, NNUNET_RAW,
                            HOLDOUT_DIR, PARTITION_DIR, require_data_dir)
from data.verify_dataset import index_tars

DS_DIR = Path(NNUNET_RAW) / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

# modality name in the tar -> nnU-Net channel index (from base.yaml: FLAIR 0, T1 1, T1ce 2, T2 3)
KIND_TO_CHANNEL = {"flair": MODALITIES["FLAIR"], "t1": MODALITIES["T1"],
                   "t1ce": MODALITIES["T1ce"], "t2": MODALITIES["T2"]}



#Block 2: Load the split
def load_split():
    f = Path(PARTITION_DIR) / "patient_split.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found, run data/split_patients.py first")
    return json.loads(f.read_text())

#Block 3: Copy one modality out of the tar
def copy_member(tar_path, member, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:") as tf, open(dest, "wb") as out:
        shutil.copyfileobj(tf.extractfile(member), out)


#Block 4: Convert the label
def write_binary_label(tar_path, member, dest, tmp_path):
    copy_member(tar_path, member, tmp_path)           # extract original seg to a temp file
    img = nib.load(str(tmp_path))
    data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(dest))
    tmp_path.unlink()


#Block 5: Convert one patient
def convert_patient(pid, files, images_dir, labels_dir):
    label_dest = labels_dir / f"{pid}.nii.gz"
    img_dests = {k: images_dir / f"{pid}_{c:04d}.nii.gz" for k, c in KIND_TO_CHANNEL.items()}
    if label_dest.exists() and all(p.exists() for p in img_dests.values()):
        return False                                   # already done
    for kind, dest in img_dests.items():
        tar_path, member = files[kind][0]
        copy_member(tar_path, member, dest)
    tar_path, member = files["seg"][0]
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_binary_label(tar_path, member, label_dest, labels_dir / f"_tmp_{pid}.nii.gz")
    return True


#Block 6: dataset.json (training set only)
def write_dataset_json(n_train):
    meta = {
        "channel_names": {str(c): k.upper() if k != "t1ce" else "T1ce"
                          for k, c in KIND_TO_CHANNEL.items()},
        "labels": {"background": 0, "WT": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    (DS_DIR / "dataset.json").write_text(json.dumps(meta, indent=2))


#Block 7: main()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="patients per split (test run)")
    args = ap.parse_args()

    split = load_split()
    patients, _, _ = index_tars(require_data_dir())

    targets = {
        "train": (DS_DIR / "imagesTr", DS_DIR / "labelsTr"),
        "val":   (HOLDOUT_DIR / "imagesVal", HOLDOUT_DIR / "labelsVal"),
        "test":  (HOLDOUT_DIR / "imagesTs", HOLDOUT_DIR / "labelsTs"),
    }
    for name, (img_dir, lbl_dir) in targets.items():
        ids = split[name][: args.limit]
        done = sum(convert_patient(p, patients[p], img_dir, lbl_dir) for p in ids)
        print(f"{name}: {len(ids)} patients ({done} converted, {len(ids) - done} skipped)")

    if args.limit is None:
        write_dataset_json(len(split["train"]))
        print("dataset.json written")

if __name__ == "__main__":
    main()