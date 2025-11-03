#!/usr/bin/env python3
import os, sys, argparse, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from phi2flux_deep import Phi2FluxDeep  # your deep model

# ---------- Data helpers ----------
def build_windows(data_path, Tc, horizons):
    d = np.load(data_path)
    phi = d["phi"]                           # [T, Nr, Ntheta, Ntor, 2]
    if "flux" in d: y = d["flux"]           # [T,3]
    else:
        Qi, Qe, Gamma = d["Qi"], d["Qe"], d["Gamma"]
        y = np.stack([Qi, Qe, Gamma], axis=-1)

    T = phi.shape[0]
    Hmax = max(horizons)
    starts = np.arange(T - Tc - Hmax + 1, dtype=int)
    if starts.size <= 0:
        raise ValueError(f"No windows: T={T}, Tc={Tc}, Hmax={Hmax}")

    X, Y = [], []
    for s in starts:
        x = phi[s:s+Tc, ...]                  # [Tc, Nr, Ntheta, Ntor, 2]
        x = np.transpose(x, (0,4,1,2,3))      # [Tc, 2, Nr, Ntheta, Ntor]
        yy = np.stack([y[s+Tc+h-1] for h in horizons], axis=0)  # [H,3]
        X.append(x); Y.append(yy)
    X = np.asarray(X, np.float32)             # [N, Tc, 2, R, Th, Tor]
    Y = np.asarray(Y, np.float32)             # [N, H, 3]
    return X, Y

def split_windows(X, Y, split=(0.6,0.2,0.2), seed=0):
    rng = np.random.default_rng(seed)
    N = len(X)
    idx = np.arange(N); rng.shuffle(idx)
    n_tr = max(1, int(split[0]*N))
    n_va = max(1, int(split[1]*N))
    if n_tr + n_va >= N:
        n_tr = max(1, N-2); n_va = 1
    n_te = N - n_tr - n_va
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    return (X[i_tr], Y[i_tr]), (X[i_va], Y[i_va]), (X[i_te], Y[i_te])

class SimpleDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, y_mean=None, y_std=None):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.y_mean = None if y_mean is None else y_mean.astype(np.float32)
        self.y_std  = None if y_std  is None else y_std.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ---------- Save/load helpers ----------
def load_nn(nn_path):
    payload = torch.load(nn_path, map_location="cpu")
    return payload

def build_model_from_payload(payload, Tc=None, horizons=None, device="cpu"):
    arch = payload["arch"]
    Tc = Tc if Tc is not None else arch["Tc"]
    horizons = horizons if horizons is not None else payload["train_horizons"]
    model = Phi2FluxDeep(
        Tc=Tc,
        horizons=horizons,
        base_channels=arch.get("base_channels", 32),
        depth=arch.get("depth", 8),
        tcn_channels=arch.get("tcn_channels", 128),
        tcn_blocks=arch.get("tcn_blocks", 3),
        dropout=arch.get("dropout", 0.0),
        norm=arch.get("norm", "bn"),
    ).to(device)
    return model

def load_partial_state(model, state_dict):
    # load everything except the horizon-specific heads.*
    filtered = {k:v for k,v in state_dict.items() if not k.startswith("heads.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected

def save_nn(nn_path, model, args_arch, train_horizons, y_mean, y_std, meta=None):
    state = model.state_dict()
    payload = {
        "state_dict": state,
        "arch": args_arch,
        "train_horizons": list(train_horizons),
        "y_mean": None if y_mean is None else y_mean,
        "y_std":  None if y_std  is None else y_std,
        "meta": meta or {}
    }
    torch.save(payload, nn_path)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="merged NPZ (phi+flux)")
    ap.add_argument("--nn_path", default="./mnt/data/bin.cgyro.nn")
    ap.add_argument("--Tc", type=int, default=64)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1,5,10],
                    help="Horizons for INFERENCE (heads will be created accordingly)")
    ap.add_argument("--more_epochs", type=int, default=0,
                    help="If >0, resume training for this many epochs; if 0, only infer")
    ap.add_argument("--train_horizons", type=int, nargs="+", default=None,
                    help="Optional override for training horizons on resume; default = payload horizons")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_dir", default="./mnt/data/myrun_logs_deep_resume")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_npz", default="./mnt/data/infer_preds_resume.npz",
                    help="Where to save inference predictions")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------- Case A: no saved NN ----------
    if not os.path.exists(args.nn_path):
        print(f"[info] No NN file at {args.nn_path}. Fresh start...")
        # Build model from CLI arch (defaults)
        model = Phi2FluxDeep(
            Tc=args.Tc, horizons=args.horizons,
            base_channels=32, depth=8,
            tcn_channels=128, tcn_blocks=3,
            dropout=args.dropout, norm="bn"
        ).to(device)
        if args.more_epochs <= 0:
            raise FileNotFoundError("No saved NN; specify --more_epochs > 0 to train, "
                                    "or run your trainer first to produce bin.cgyro.nn.")
        # Fall through to training branch with empty payload
        payload = {
            "arch": {"Tc": args.Tc, "base_channels":32,"depth":8,
                     "tcn_channels":128,"tcn_blocks":3,"dropout":args.dropout,"norm":"bn"},
            "train_horizons": list(args.horizons),
            "y_mean": None, "y_std": None,
        }
    else:
        # ---------- Case B: load saved NN ----------
        payload = load_nn(args.nn_path)
        print(f"[load] Found NN @ {args.nn_path}")
        # Build model for INFERENCE horizons first (we'll reload heads for training if needed)
        model = build_model_from_payload(payload, Tc=args.Tc, horizons=args.horizons, device=device)
        missing, unexpected = load_partial_state(model, payload["state_dict"])
        print(f"[load] partial load: missing={len(missing)} (heads expected), unexpected={len(unexpected)}")

    # ---------- If we need to resume training ----------
    if args.more_epochs > 0:
        train_hz = args.train_horizons if args.train_horizons is not None else payload["train_horizons"]
        if train_hz is None:
            # If payload had no train_horizons (fresh), use inference horizons for training too
            train_hz = args.horizons
        print(f"[train] horizons for training: {train_hz}")

        # Rebuild model WITH TRAINING HORIZONS (so heads match during resume)
        model = build_model_from_payload(payload, Tc=args.Tc, horizons=train_hz, device=device)
        # Load full state (including heads) when shapes match; else load partial
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
            print("[train] loaded full state (heads matched).")
        except Exception as e:
            print(f"[train] heads mismatch; loading shared layers only. ({e})")
            load_partial_state(model, payload["state_dict"])

        # Build windows for TRAIN horizons
        X, Y = build_windows(args.data, args.Tc, train_hz)
        (Xtr, Ytr), (Xv, Yv), (Xte, Yte) = split_windows(X, Y, split=(0.6,0.2,0.2), seed=args.seed)
        # Standardize Y using TRAIN set statistics
        y_mean = Ytr.mean(axis=(0,1), keepdims=True); y_std = Ytr.std(axis=(0,1), keepdims=True) + 1e-8
        Ytr_n, Yv_n = (Ytr - y_mean)/y_std, (Yv - y_mean)/y_std

        train_ds = SimpleDS(Xtr, Ytr_n, y_mean, y_std)
        val_ds   = SimpleDS(Xv,  Yv_n,  y_mean, y_std)

        crit = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        os.makedirs(args.log_dir, exist_ok=True)
        best_val = float("inf")
        start = time.time()
        for ep in range(1, args.more_epochs+1):
            # ---- train ----
            model.train()
            t0 = time.time()
            tr_sum, tr_n = 0.0, 0
            loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
                tr_sum += loss.item()*len(xb); tr_n += len(xb)

            # ---- val ----
            model.eval()
            va_sum, va_n = 0.0, 0
            with torch.no_grad():
                for xb, yb in DataLoader(val_ds, batch_size=args.batch, shuffle=False):
                    xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                    yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
                    pred = model(xb)
                    loss = crit(pred, yb)
                    va_sum += loss.item()*len(xb); va_n += len(xb)

            tr = tr_sum/max(1,tr_n); va = va_sum/max(1,va_n)
            print(f"[resume] epoch {ep:03d}/{args.more_epochs}  "
                  f"train={tr:.4f} val={va:.4f}  ({time.time()-t0:.1f}s)")

            # save best back into nn_path
            if va < best_val - 1e-4:
                best_val = va
                # save consolidated NN
                args_arch = {
                    "Tc": args.Tc,
                    "base_channels": 32,
                    "depth": 8,
                    "tcn_channels": 128,
                    "tcn_blocks": 3,
                    "dropout": args.dropout,
                    "norm": "bn",
                }
                save_nn(args.nn_path, model, args_arch, train_hz,
                        y_mean=y_mean, y_std=y_std,
                        meta={"val_loss": float(va), "resumed_epochs": ep})
                print(f"[save] updated {args.nn_path} (val={va:.4f})")
        print(f"[resume] done in {(time.time()-start)/60:.1f} min")

        # After resume, if you also want inference on NEW horizons, fall through:
        # the code below will rebuild inference heads and predict.

    # ---------- Inference on (possibly NEW) horizons ----------
    # Rebuild model with INFERENCE horizons and load shared layers
    model = build_model_from_payload(payload, Tc=args.Tc, horizons=args.horizons, device=device)
    try:
        # try to load full if horizons match saved heads
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception:
        load_partial_state(model, payload["state_dict"])
    model.eval()

    # Choose scalers for de-normalization:
    y_mean = payload.get("y_mean", None)
    y_std  = payload.get("y_std", None)

    X, Y = build_windows(args.data, args.Tc, args.horizons)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            xb = torch.from_numpy(X[i:i+args.batch]).to(device=device, dtype=torch.float32)
            yb = model(xb).cpu().numpy()  # [B, H, 3]
            preds.append(yb)
    P = np.concatenate(preds, 0)

    if y_mean is not None and y_std is not None:
        # ensure float32 arrays
        y_mean = y_mean.astype(np.float32); y_std = y_std.astype(np.float32)
        P_phys = P * y_std + y_mean
    else:
        print("[warn] No y_mean/y_std in NN payload; predictions in normalized units.")
        P_phys = P

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz,
        preds=P_phys,
        horizons=np.array(args.horizons, dtype=np.int32),
        Tc=np.array([args.Tc], dtype=np.int32),
    )
    print(f"[infer] saved predictions → {args.out_npz}")

if __name__ == "__main__":
    main()

