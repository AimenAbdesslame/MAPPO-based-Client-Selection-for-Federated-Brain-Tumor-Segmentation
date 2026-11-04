
#Block 1: imports. Reuse the indexer rather than re-reading the tars in a new way
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import SEED, N_TRAIN, N_VAL, N_TEST, PARTITION_DIR, require_data_dir
from data.verify_dataset import index_tars, REQUIRED



#Block 2: get the pool. Complete patients only, sorted so the result doesn’t depend on tar order.
def get_pool():
    patients, _, _ = index_tars(require_data_dir())
    return sorted(p for p, f in patients.items() if REQUIRED <= set(f))



#Block 3: split. Shuffle with a local random.Random(SEED), take the first 400, and cut it up.
def split(pool):
    n = N_TRAIN + N_VAL + N_TEST
    assert len(pool) >= n, f"pool has {len(pool)} patients, need {n}"
    chosen = random.Random(SEED).sample(pool, n)
    return {
        "train": sorted(chosen[:N_TRAIN]),
        "val": sorted(chosen[N_TRAIN:N_TRAIN + N_VAL]),
        "test": sorted(chosen[N_TRAIN + N_VAL:]),
    }


#Block 4: checks. This is the “proof of no overlap”
def check(splits):
    sets = {k: set(v) for k, v in splits.items()}
    assert len(sets["train"] & sets["val"]) == 0
    assert len(sets["train"] & sets["test"]) == 0
    assert len(sets["val"] & sets["test"]) == 0
    assert all(len(sets[k]) == len(splits[k]) for k in splits), "duplicate IDs inside a split"


#Block 5: save. Store metadata too (seed, pool size, counts) so the manifest explains itself.
def save(splits, pool_size):
    manifest = {"seed": SEED, "pool_size": pool_size,
                "counts": {k: len(v) for k, v in splits.items()}, **splits}
    out = Path(PARTITION_DIR) / "patient_split.json"
    if out.exists():
        raise FileExistsError(f"{out} exists, delete it deliberately to regenerate")
    out.write_text(json.dumps(manifest, indent=2))
    return out


#Block 6: main()
def main():
    pool = get_pool()
    splits = split(pool)
    check(splits)
    print(f"pool={len(pool)}", {k: len(v) for k, v in splits.items()})
    print("saved:", save(splits, len(pool)))

if __name__ == "__main__":
    main()

