#Block 1: Imports and constants
import argparse, json, random, re, sys, tarfile, tempfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import SEED, RESULTS_DIR, require_data_dir

# matches both layouts: ./BraTS2021_00000/BraTS2021_00000_flair.nii.gz and ./BraTS2021_00495_flair.nii.gz
PATTERN = re.compile(r"(BraTS2021_\d+)_(flair|t1ce|t1|t2|seg)\.nii\.gz$")
REQUIRED = {"flair", "t1", "t1ce", "t2", "seg"}
VALID_LABELS = {0, 1, 2, 4}


#Block 2: Index the tars (no extraction)
def index_tars(data_dir):
    patients = defaultdict(lambda: defaultdict(list))
    unexpected = []
    for tar_path in sorted(Path(data_dir).glob("*.tar")):
        with tarfile.open(tar_path, "r:") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                match = PATTERN.search(m.name)
                if match:
                    pid, kind = match.groups()
                    patients[pid][kind].append((tar_path, m.name))
                else:
                    unexpected.append(f"{tar_path.name}:{m.name}")
    return patients, unexpected



#Block 3: Structure checks (all patients)
def check_structure(patients):
    issues = []
    for pid, files in sorted(patients.items()):
        missing = REQUIRED - set(files)
        if missing:
            issues.append(f"{pid}: missing {sorted(missing)}")
        for kind, entries in files.items():
            if len(entries) > 1:
                issues.append(f"{pid}: duplicate '{kind}' in {[e[0].name for e in entries]}")
    return issues



#Block 4: Extract a sample to a temp folder
def extract_patient(files, out_dir):
    """files = patients[pid]; writes the 5 files to out_dir, returns {kind: path}."""
    paths = {}
    for kind in REQUIRED:
        tar_path, member = files[kind][0]
        dest = Path(out_dir) / f"{kind}.nii.gz"
        with tarfile.open(tar_path, "r:") as tf, open(dest, "wb") as out:
            out.write(tf.extractfile(member).read())
        paths[kind] = dest
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return paths


#Block 5: Deep NIfTI check (one patient)
def check_patient(pid, paths):
    issues = []
    try:
        imgs = {k: nib.load(str(p)) for k, p in paths.items()}
        seg = np.asarray(imgs["seg"].dataobj)
    except Exception as e:
        return [f"{pid}: unreadable NIfTI ({e})"], None

    ref = imgs["seg"]
    for kind, img in imgs.items():
        if img.shape != ref.shape:
            issues.append(f"{pid}: shape mismatch {kind} {img.shape} vs seg {ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-3):
            issues.append(f"{pid}: affine mismatch {kind}")

    labels = set(np.unique(seg).astype(int).tolist())
    if not labels <= VALID_LABELS:
        issues.append(f"{pid}: unexpected labels {sorted(labels - VALID_LABELS)}")

    stats = {"shape": list(ref.shape), "wt_voxels": int((seg > 0).sum())}
    return issues, stats


#Block 6: Report
def save_report(report):
    out = Path(RESULTS_DIR) / "verify_report.json"
    out.write_text(json.dumps(report, indent=2))
    return out


#Block 7: main()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20, help="patients to deep-check")
    args = ap.parse_args()

    data_dir = require_data_dir()
    patients, unexpected = index_tars(data_dir)
    print(f"Patients indexed: {len(patients)}")

    issues = check_structure(patients)

    complete = sorted(p for p, f in patients.items() if REQUIRED <= set(f))
    sample = random.Random(SEED).sample(complete, min(args.sample, len(complete)))
    sample_stats = {}
    with tempfile.TemporaryDirectory() as tmp:
        for pid in sample:
            paths = extract_patient(patients[pid], Path(tmp) / pid)  # make sure the folder exists
            p_issues, stats = check_patient(pid, paths)
            issues += p_issues
            if stats:
                sample_stats[pid] = stats

    report = {
        "n_patients": len(patients),
        "n_complete": len(complete),
        "n_unexpected_files": len(unexpected),
        "unexpected_files": unexpected[:20],
        "sample_size": len(sample),
        "n_issues": len(issues),
        "issues": issues,
        "sample_stats": sample_stats,
    }
    print(f"Report: {save_report(report)}")
    print(f"Issues: {len(issues)}")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()


