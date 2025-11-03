#!/usr/bin/env python3
# Use the deep model + a self-contained build_datasets
import os, sys, types, numpy as np, torch
sys.path.append(os.path.dirname(__file__))

from phi2flux_deep import Phi2FluxDeep as Phi2Flux  # model alias the trainer expects

def build_datasets(data_path, Tc, horizons, split=(0.6, 0.2, 0.2), seed=0):
    """
    Loads merged NPZ with:
      - phi:  [T, Nr, Ntheta, Ntor, 2]
      - flux: [T, 3] (Qi, Qe, Gamma)
    Creates sliding windows:
      X: [N, Tc, 2, Nr, Ntheta, Ntor]
      Y: [N, H, 3]  targets at t+Δ for Δ in horizons
    Standardizes Y using training split stats (channel-wise).
    """
    d = np.load(data_path)
    phi = d["phi"]      # [T, Nr, Ntheta, Ntor, 2]
    if "flux" in d:
        y = d["flux"]  # [T,3]
    else:
        Qi, Qe, Gamma = d["Qi"], d["Qe"], d["Gamma"]
        y = np.stack([Qi, Qe, Gamma], axis=-1)  # [T,3]

    T = phi.shape[0]
    Hs = list(horizons)
    Hmax = max(Hs)

    # ---- sliding-window starts ----
    starts = np.arange(T - Tc - Hmax + 1, dtype=int)
    if starts.size <= 0:
        raise ValueError(
            f"No training windows: T={T}, Tc={Tc}, max(horizon)={Hmax} → T-Tc-Hmax+1 <= 0.\n"
            f"Pick smaller Tc / horizons, or use a longer time series."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(starts)

    # ---- create 60/20/20 with non-empty guard ----
    n = len(starts)
    n_tr = max(1, int(split[0]*n))
    n_val = max(1, int(split[1]*n))
    if n_tr + n_val >= n:
        n_tr = max(1, n-2)
        n_val = 1
    n_te = max(1, n - n_tr - n_val)
    idx_tr = starts[:n_tr]
    idx_val = starts[n_tr:n_tr+n_val]
    idx_te  = starts[n_tr+n_val:n_tr+n_val+n_te]

    def make_set(idxs):
        X, Y = [], []
        for s in idxs:
            x = phi[s:s+Tc, ...]                 # [Tc, Nr, Ntheta, Ntor, 2]
            x = np.transpose(x, (0,4,1,2,3))     # [Tc, 2, Nr, Ntheta, Ntor]
            yy = np.stack([y[s+Tc+h-1] for h in Hs], axis=0)  # [H,3]
            X.append(x); Y.append(yy)
        return np.stack(X), np.stack(Y)

    Xtr, Ytr = make_set(idx_tr)
    Xv,  Yv  = make_set(idx_val)
    Xte, Yte = make_set(idx_te)

    # ---- standardize Y using TRAIN stats ----
    y_mean = Ytr.mean(axis=(0,1), keepdims=True)
    y_std  = Ytr.std(axis=(0,1), keepdims=True) + 1e-8
    Ytr_n, Yv_n, Yte_n = (Ytr - y_mean)/y_std, (Yv - y_mean)/y_std, (Yte - y_mean)/y_std

    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i): return self.X[i], self.Y[i]

    # small print so you can see split sizes
    print(f"[build_datasets] windows: total={n}  train={len(Xtr)}  val={len(Xv)}  test={len(Xte)}")

    # bake means into attributes the trainer can read if needed
    ds_tr = SimpleDS(Xtr, Ytr_n); ds_tr.y_mean = y_mean.astype(np.float32); ds_tr.y_std = y_std.astype(np.float32)
    ds_va = SimpleDS(Xv,  Yv_n ); ds_va.y_mean = y_mean.astype(np.float32); ds_va.y_std = y_std.astype(np.float32)
    ds_te = SimpleDS(Xte, Yte_n); ds_te.y_mean = y_mean.astype(np.float32); ds_te.y_std = y_std.astype(np.float32)
    return ds_tr, ds_va, ds_te

# ---- Inject our model + datasets under the name the trainer imports ----
shim = types.ModuleType("phi2flux_3species")
shim.Phi2Flux = Phi2Flux
shim.build_datasets = build_datasets
sys.modules["phi2flux_3species"] = shim

# ---- Now run the improved trainer (which will import phi2flux_3species) ----
import train_with_earlystop_rmse  # executes main on import

