
#Block 1: setup
import os, sys, time, torch
from pathlib import Path

os.environ["nnUNet_raw"]          = "/kaggle/working/nnunet/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnunet/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = "/kaggle/working/nnunet/nnUNet_results"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.nnunet_adapter import (write_client_splits, build_trainer, set_client,
                                    get_weights, set_weights, train_local)

write_client_splits()



#Block 2: client switching
trainer = build_trainer(batch_size=16)
for i in range(4):
    set_client(trainer, i)
    ds_tr, _ = trainer.get_tr_and_val_datasets()
    print(f"client {i}: {len(ds_tr.identifiers)} train cases")



#Block 3: weight exchange
w0 = get_weights(trainer)

set_client(trainer, 0)
train_local(trainer, 5)
w1 = get_weights(trainer)

assert any(not torch.equal(w0[k], w1[k]) for k in w0), "weights unchanged after training"
print("weights changed after training: OK")

set_weights(trainer, w0)
w_reset = get_weights(trainer)
assert all(torch.equal(w0[k], w_reset[k]) for k in w0), "set_weights did not restore w0"
print("weights restored: OK")



#Block 4: two-client compatibility
set_client(trainer, 1)
set_weights(trainer, w0)
train_local(trainer, 5)
w2 = get_weights(trainer)
assert w2.keys() == w0.keys() and all(w2[k].shape == w0[k].shape for k in w0)
print("cross-client weight shapes: OK")



#Block 5: model facts
n_keys      = len(w0)
model_bytes = sum(v.numel() * v.element_size() for v in w0.values())
non_float   = {k: str(v.dtype) for k, v in w0.items() if not v.is_floating_point()}
print(f"keys={n_keys}  model_bytes={model_bytes/1024**2:.1f} MB  non-float={non_float}")



#Block 6: memory and speed
def _cleanup(t):
    for dl in (getattr(t, "dataloader_train", None), getattr(t, "dataloader_val", None)):
        if dl is not None and hasattr(dl, "_finish"):
            dl._finish()

for bs in (16, 32):
    _cleanup(trainer)
    del trainer
    torch.cuda.empty_cache()
    trainer = build_trainer(batch_size=bs)
    set_client(trainer, 0)
    for _ in range(2):
        trainer.train_step(next(trainer.dataloader_train))
    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        trainer.train_step(next(trainer.dataloader_train))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem_gb  = torch.cuda.max_memory_allocated() / 1024**3
    print(f"bs={bs}  peak_mem={mem_gb:.2f} GB  sec/step={elapsed/10:.3f}s")

