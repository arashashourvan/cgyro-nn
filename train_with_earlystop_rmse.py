#!/usr/bin/env python3
"""
train_with_earlystop_rmse.py
----------------------------
Improved training runner for your Phi→Flux model that *reuses* your existing code.

It imports:
  - Phi2Flux (model) and build_datasets(...) from phi2flux_3species.py

Adds:
  - Dataset diagnostics (target mean/std + histograms)
  - Early stopping (patience)
  - Per-channel RMSE (Qi, Qe, Gamma) every epoch
  - Checkpoint "best.pt" saved only on validation improvement
  - Plots: loss_curve.png, rmse_bar.png
  - Optionally report RMSE in *physical units* (de-normalized)

Usage (example):
  python train_with_earlystop_rmse.py \
    --data ./mnt/data/myrun_phi_flux.npz \
    --Tc 64 --horizons 1 2 5 \
    --epochs 80 --batch 8 --lr 3e-4 \
    --device cpu \
    --log_dir ./mnt/data/myrun_logs \
    --early_stop 10 --weight_decay 1e-4 \
    --phys_units 1
"""

import argparse, os, time, csv, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# ------------------------
# CLI
# ------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--Tc", type=int, default=64)
ap.add_argument("--horizons", type=int, nargs="+", default=[1,2,5])
#ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--log_dir", default="./logs")
ap.add_argument("--early_stop", type=int, default=10, help="early stopping patience")
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--phys_units", type=int, default=0, help="1=report extra RMSE in physical units if stats available")
ap.add_argument("--nn_path", default="./mnt/data/bin.cgyro.nn",help="Save/load unified trained NN (weights + cfg + y scalers)")

args = ap.parse_args()

os.makedirs(args.log_dir, exist_ok=True)

# ------------------------
# Import model & dataset from your training module
# ------------------------
try:
    from phi2flux_3species import Phi2Flux, build_datasets
except Exception as e:
    print("[ERROR] Could not import Phi2Flux/build_datasets from phi2flux_3species.py")
    print("Ensure this script sits next to phi2flux_3species.py.")
    print("Import error:", repr(e))
    raise

device = torch.device(args.device)

# ------------------------
# Build datasets (this should apply the same normalization used in training)
# ------------------------
train_ds, val_ds, test_ds = build_datasets(
    data_path=args.data,
    Tc=args.Tc,
    horizons=args.horizons,
    split=(0.6, 0.2, 0.2),
    seed=0
)

# ------------------------
# Diagnostics: check target scaling and save histograms
# ------------------------
def describe_flux(name, ds, outdir):
    y_mean = getattr(ds, "y_mean", None)
    y_std  = getattr(ds, "y_std", None)
    print(f"\n[{name}] dataset:")
    if y_mean is not None and y_std is not None:
        print(f"  y_mean: {np.round(y_mean, 5)}")
        print(f"  y_std : {np.round(y_std, 5)}")
    else:
        print("  (no stored normalization stats)")

    n = min(len(ds), 200)
    Y = []
    for i in range(n):
        _, y = ds[i]
        Y.append(y)
    Y = np.stack(Y) if len(Y) else np.zeros((1, len(args.horizons), 3))
    print(f"  sample Y: min={Y.min():.3e}, max={Y.max():.3e}, mean={Y.mean():.3e}, std={Y.std():.3e}")

    plt.figure(figsize=(6,3))
    try:
        plt.hist(Y[...,0].ravel(), bins=50, alpha=0.5, label="Qi")
        plt.hist(Y[...,1].ravel(), bins=50, alpha=0.5, label="Qe")
        plt.hist(Y[...,2].ravel(), bins=50, alpha=0.5, label="Gamma")
    except Exception:
        pass
    plt.title(f"{name} flux distributions (first {n} samples, standardized)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_flux_hist.png"), dpi=160)
    plt.close()

describe_flux("train", train_ds, args.log_dir)
describe_flux("val",   val_ds,   args.log_dir)
describe_flux("test",  test_ds,  args.log_dir)
print("✅ Saved flux histograms in", args.log_dir)

# ------------------------
# Model, loss, optimizer
# ------------------------
model = Phi2Flux(Tc=args.Tc, horizons=args.horizons, dropout=args.dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ------------------------
# Helpers
# ------------------------
def rmse_per_channel(y_true, y_pred):
    """Compute per-channel RMSE over all horizons and samples in standardized space."""
    diff = y_true - y_pred
    mse = (diff ** 2).mean(axis=(0,1))  # mean over N,H -> [3]
    return np.sqrt(mse)

def rmse_per_channel_physical(y_true, y_pred, y_std):
    """Convert standardized RMSE back to physical units per channel if y_std provided."""
    # RMSE_std * std_phys  (assuming y was standardized by (y - mean)/std)
    return rmse_per_channel(y_true, y_pred) * y_std.squeeze()

# ------------------------
# Training loop with early stopping + per-channel RMSE
# ------------------------
best_val = np.inf
epochs_no_improve = 0
history = []

for epoch in range(1, args.epochs + 1):
    t_epoch = time.time()
    # print(f"\n🟦 Starting epoch {epoch}/{args.epochs} ...", flush=True)
    # ---- train ----
    model.train()
    n_steps = math.ceil(len(train_ds)/args.batch)
    print(f"🟦 Starting epoch {epoch}/{args.epochs} (train steps ≈ {n_steps})", flush=True)
    t0 = time.time()
    train_loss = 0.0
    n_train = 0
    for step, (xb,yb) in enumerate(DataLoader(train_ds, batch_size=args.batch, shuffle=True)):
    #    xb = torch.tensor(xb, dtype=torch.float32, device=device)
    #    yb = torch.tensor(yb, dtype=torch.float32, device=device)
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        if torch.isnan(loss):
            print(f"❗ NaN loss at epoch {epoch} step {step}", flush=True)
            break
        loss.backward()
        optimizer.step()

        if step % 2 == 0:  # heartbeat every ~10 steps
            dt = time.time() - t0
            print(f"  ... step {step}/{n_steps}  loss={loss.item():.4f}  ({dt:.1f}s since last print)",
                  flush=True)
            t0 = time.time()
            total_norm = 0.0
            for p in model.parameters():
               if p.grad is not None:
                   param_norm = p.grad.data.norm(2).item()
                   total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            print(f"  ↘ grad L2 norm ≈ {total_norm:.3e} (batch {step})", flush=True)

        train_loss += loss.item() * len(xb)
        n_train += len(xb)
    train_loss /= max(1, n_train)

    print(f"✅ epoch {epoch} forward/backward done in {(time.time()-t_epoch):.1f}s", flush=True)

    # ---- validation ----
    model.eval()
    val_loss = 0.0
    n_val = 0
    all_true, all_pred = [], []
    #v_steps = math.ceil(len(val_ds)/args.batch)
    with torch.no_grad():
        for vstep, (xb, yb) in enumerate(DataLoader(val_ds, batch_size=args.batch, shuffle=False)):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.float32, device=device)
            pred = model(xb)
            vloss = criterion(pred, yb)  
            val_loss += criterion(pred, yb).item() * len(xb)
            if vstep == 0:
                print(f"  🔎 val step 0: batch {tuple(xb.shape)} loss={vloss.item():.4f}", flush=True)
            n_val += len(xb)
            all_true.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    val_loss /= max(1, n_val)

    Y = np.concatenate(all_true, 0) if all_true else np.zeros((1, len(args.horizons), 3))
    P = np.concatenate(all_pred, 0) if all_pred else np.zeros_like(Y)
    rmse_ch = rmse_per_channel(Y, P)

    # Optional: report RMSE in physical units using validation stats
    rmse_phys = None
    if args.phys_units and getattr(val_ds, "y_std", None) is not None:
        rmse_phys = rmse_per_channel_physical(Y, P, val_ds.y_std)
        print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"RMSE(std)=[Qi:{rmse_ch[0]:.3f}, Qe:{rmse_ch[1]:.3f}, Γ:{rmse_ch[2]:.3f}]  "
              f"RMSE(phys)=[Qi:{rmse_phys[0]:.3f}, Qe:{rmse_phys[1]:.3f}, Γ:{rmse_phys[2]:.3f}]  "
              f"({time.time()-t0:.1f}s)")
    else:
        print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"rmse=[Qi:{rmse_ch[0]:.3f}, Qe:{rmse_ch[1]:.3f}, Γ:{rmse_ch[2]:.3f}]  "
              f"({time.time()-t0:.1f}s)")

    # Checkpoint on improvement
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "rmse_ch": rmse_ch,
            "cfg": vars(args)
        }, os.path.join(args.log_dir, "best.pt"))
    else:
        epochs_no_improve += 1

    history.append([epoch, train_loss, val_loss, *rmse_ch])

    # ---- Also save a unified production file for inference/resume ----
    # If you use DDP, prefer model.module.state_dict()
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    # Try to capture y scalers from your training dataset if available
    def _get_np(a):
        import numpy as _np
        if a is None: return None
        try:
            return a.detach().cpu().numpy()
        except Exception:
            try:
                return a.cpu().numpy()
            except Exception:
                return _np.array(a)

    y_mean = getattr(train_ds, "y_mean", None)
    y_std  = getattr(train_ds, "y_std", None)
    y_mean = _get_np(y_mean)
    y_std  = _get_np(y_std)

    payload = {
        "state_dict": state,
        "arch": {
            "Tc": args.Tc,
            "base_channels": getattr(args, "base_channels", 32),
            "depth": getattr(args, "depth", 8),
            "tcn_channels": getattr(args, "tcn_channels", 128),
            "tcn_blocks": getattr(args, "tcn_blocks", 3),
            "dropout": getattr(args, "dropout", 0.0),
            "norm": getattr(args, "norm", "bn"),
        },
        "train_horizons": list(args.horizons),
        "y_mean": y_mean,
        "y_std": y_std,
        "meta": {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "log_dir": args.log_dir,
        },
    }
    import os, torch
    os.makedirs(os.path.dirname(args.nn_path), exist_ok=True)
    torch.save(payload, args.nn_path)
    print(f"✅ Saved unified NN → {args.nn_path}")


    if epochs_no_improve >= args.early_stop:
        print(f"⏹ Early stopping after {epoch} epochs (no val improvement).")
        break


# ------------------------
# Save metrics and plots
# ------------------------
csv_path = os.path.join(args.log_dir, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "rmse_Qi", "rmse_Qe", "rmse_Gamma"])
    writer.writerows(history)
print("Saved metrics →", csv_path)

hist_arr = np.array(history) if len(history) else np.zeros((1,6))
plt.figure(figsize=(6,4))
plt.plot(hist_arr[:,0], hist_arr[:,1], label="train")
plt.plot(hist_arr[:,0], hist_arr[:,2], label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, "loss_curve.png"), dpi=160)
plt.close()

labels = ["Qi","Qe","Gamma"]
plt.figure(figsize=(5,4))
for i,lbl in enumerate(labels):
    plt.plot(hist_arr[:,0], hist_arr[:,3+i], label=lbl)
plt.xlabel("epoch"); plt.ylabel("RMSE (std units)") ; plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, "rmse_bar.png"), dpi=160)
plt.close()

# Final write of the unified NN (re-uses last/best model on rank 0)
try:
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "state_dict": state,
        "arch": {
            "Tc": args.Tc,
            "base_channels": getattr(args, "base_channels", 32),
            "depth": getattr(args, "depth", 8),
            "tcn_channels": getattr(args, "tcn_channels", 128),
            "tcn_blocks": getattr(args, "tcn_blocks", 3),
            "dropout": getattr(args, "dropout", 0.0),
            "norm": getattr(args, "norm", "bn"),
        },
        "train_horizons": list(args.horizons),
        "y_mean": _get_np(getattr(train_ds, "y_mean", None)),
        "y_std":  _get_np(getattr(train_ds, "y_std", None)),
        "meta": {"final_write": True, "best_val": float(best_val)},
    }
    import os, torch
    os.makedirs(os.path.dirname(args.nn_path), exist_ok=True)
    torch.save(payload, args.nn_path)
    print(f"✅ Final unified NN → {args.nn_path}")
except Exception as e:
    print(f"[warn] final save_nn skipped: {e}")


# Save summary JSON
summary = {
    "best_val": float(best_val),
    "epochs_ran": int(hist_arr[-1,0]) if len(history) else 0,
    "Tc": args.Tc, "horizons": args.horizons,
    "batch": args.batch, "lr": args.lr,
    "early_stop": args.early_stop, "weight_decay": args.weight_decay,
    "dropout": args.dropout
}
with open(os.path.join(args.log_dir, "train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Training complete. Artifacts in:", args.log_dir)
