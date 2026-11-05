
#Block 1: imports and load
import json, random, sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (SEED, N_TRAIN, PARTITION_DIR, NNUNET_RAW,
                             CLIENT_NAMES, CLIENT_PROPORTIONS, CLIENT_COSTS, CFG)

_mix   = CFG["clients"]["quartile_mix"]          # 4×4 weights from base.yaml
_split = json.loads((Path(PARTITION_DIR) / "patient_split.json").read_text())
TRAIN  = _split["train"]
assert len(TRAIN) == N_TRAIN



#Block 2: tumor volume per patient (reads converted labelsTr)
def volumes(pids):
    labels_dir = Path(NNUNET_RAW) / f"Dataset{CFG['dataset']['dataset_id']:03d}_{CFG['dataset']['dataset_name']}" / "labelsTr"
    return {pid: int(np.asarray(nib.load(str(next(labels_dir.glob(f"{pid}*")))).dataobj).astype(bool).sum())
            for pid in pids}



#Block 3: client sizes (round proportions, fix last so sum == N_TRAIN)
def client_sizes():
    sizes = [round(p * N_TRAIN) for p in CLIENT_PROPORTIONS]
    sizes[-1] = N_TRAIN - sum(sizes[:-1])
    return sizes



#Block 4: quartile-mix assignment
def assign(pids, vols, sizes):
    order  = sorted(pids, key=lambda p: vols[p])
    n      = len(order)
    q_bins = [order[i * n // 4 : (i + 1) * n // 4] for i in range(4)]
    rng    = random.Random(SEED)
    result = {}
    used   = set()
    for name, size, weights in zip(CLIENT_NAMES, sizes, _mix):
        pool = []
        for bucket, w in zip(q_bins, weights):
            avail = [p for p in bucket if p not in used and p not in pool]
            pool += rng.sample(avail, min(round(w * size), len(avail)))
        remaining = [p for p in order if p not in used and p not in pool]
        need = size - len(pool)
        if need > 0:
            pool += rng.sample(remaining, min(need, len(remaining)))
        pool = pool[:size]
        used.update(pool)
        result[name] = sorted(pool)
    return result



#Block 5: checks
def check(partitions, vols):
    sizes  = dict(zip(CLIENT_NAMES, client_sizes()))
    all_ids = set(TRAIN)
    assigned = set()
    for name, pids in partitions.items():
        s = set(pids)
        assert not s & assigned,       f"Client {name} overlaps with another"
        assert s <= all_ids,           f"Client {name} has IDs outside train set"
        assert len(pids) == sizes[name], f"Client {name}: got {len(pids)}, expected {sizes[name]}"
        assigned |= s
    assert assigned == all_ids, f"Not all train IDs assigned ({len(assigned)}/{len(all_ids)})"



#Block 6: stats report
def stats(partitions, vols):
    order = sorted(vols, key=lambda p: vols[p])
    n     = len(order)
    bins  = [set(order[i * n // 4 : (i + 1) * n // 4]) for i in range(4)]
    report = {}
    for name, cost in zip(CLIENT_NAMES, CLIENT_COSTS):
        v  = [vols[p] for p in partitions[name]]
        ps = set(partitions[name])
        report[name] = {
            "n": len(v), "cost_multiplier": cost,
            "wt_voxels": {"mean": float(np.mean(v)), "median": float(np.median(v)), "std": float(np.std(v))},
            "quartile_counts": [len(ps & b) for b in bins],   # realised Q1..Q4 counts
        }
    return report



#Block 7: save
def save(partitions, vols):
    out = Path(PARTITION_DIR) / "client_partitions.json"
    if out.exists():
        raise FileExistsError(f"{out} exists, delete it deliberately to regenerate")
    payload = {"seed": SEED, "stats": stats(partitions, vols),
               "counts": {n: len(p) for n, p in partitions.items()}, **partitions}
    out.write_text(json.dumps(payload, indent=2))
    return out



#Block 8: main
def main():
    vols       = volumes(TRAIN)
    sizes      = client_sizes()
    partitions = assign(TRAIN, vols, sizes)
    check(partitions, vols)
    print({n: len(p) for n, p in partitions.items()})
    print("saved:", save(partitions, vols))

if __name__ == "__main__":
    main()

