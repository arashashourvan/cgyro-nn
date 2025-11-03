
import argparse, time, sys, json
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import os

# ----------------------- Dataset & Windowing -----------------------

@dataclass
class WindowSpec:
    Tc: int
    horizons: List[int]
    stride: int = 1

class PhiFluxDataset(Dataset):
    def __init__(self, phi: np.ndarray, flux: np.ndarray, spec: WindowSpec,
                 t_range: Tuple[int,int], standardize: bool = True):
        assert phi.ndim == 5 and phi.shape[-1] == 2, "phi must be [T,KX,KY,TH,2]"
        assert flux.ndim == 2 and flux.shape[1] == 3, "flux must be [T,3]=[Qi,Qe,Gamma]"
        self.spec = spec
        self.phi = phi.astype(np.float32)
        self.flux = flux.astype(np.float32)
        self.T = phi.shape[0]
        self.t0, self.t1 = t_range
        max_h = max(spec.horizons)
        starts = list(range(self.t0, self.t1+1, spec.stride))
        self.starts = [t for t in starts if t + spec.Tc - 1 + max_h < self.T]
        if standardize:
            x = self.phi[self.t0:self.t1+spec.Tc]
            self.x_mean = x.mean(axis=(0,1,2,3), keepdims=True)
            self.x_std  = x.std(axis=(0,1,2,3), keepdims=True) + 1e-6
            y = self.flux[self.t0:self.t1+spec.Tc]
            self.y_mean = y.mean(axis=0, keepdims=True)
            self.y_std  = y.std(axis=0, keepdims=True) + 1e-6
        else:
            self.x_mean = 0.0; self.x_std = 1.0
            self.y_mean = 0.0; self.y_std = 1.0

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        t = self.starts[idx]
        Tc = self.spec.Tc
        x = self.phi[t:t+Tc]
        x = (x - self.x_mean)/self.x_std
        ys = []
        for h in self.spec.horizons:
            y = self.flux[t+Tc-1+h]
            y = (y - self.y_mean[0]) / self.y_std[0]
            ys.append(y)
        y = np.stack(ys, axis=0)
        x = np.transpose(x, (0,4,1,2,3)).astype(np.float32)  # [Tc,2,KX,KY,TH]
        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------- Model: 3D CNN + causal TCN -----------------------

class Spatial3DEncoder(nn.Module):
    def __init__(self, c_in=2, c_mid=32, c_out=64):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(c_mid, c_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B,2,KX,KY,TH]
        import torch.nn.functional as F
        B,C,KX,KY,TH = x.shape
        z = self.act(self.conv1(x))
        z = self.act(self.conv2(z))
        # Dynamic spatial pooling to handle small dims (e.g., KY=1)
        tgt1 = (min(16, KX), min(16, KY), min(4, TH))
        z = F.adaptive_avg_pool3d(z, tgt1)
        z = self.act(self.conv3(z))
        tgt2 = (min(4, tgt1[0]), min(4, tgt1[1]), min(2, tgt1[2]))
        z = F.adaptive_avg_pool3d(z, tgt2)
        return z.flatten(1)

class TemporalTCN(nn.Module):
    def __init__(self, f_in, f_hid=128, levels=4, horizons=1, n_out=3, dropout=0.1):
        super().__init__()
        layers = []
        c = f_in
        for i in range(levels):
            d = 2**i
            layers += [
                nn.Conv1d(c, f_hid, kernel_size=3, padding=d, dilation=d),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(f_hid, f_hid, kernel_size=3, padding=d, dilation=d),
                nn.GELU(), nn.Dropout(dropout),
            ]
            c = f_hid
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(c, horizons*n_out, kernel_size=1)
        self.horizons = horizons; self.n_out = n_out

    def forward(self, z_seq):                 # z_seq: [B,T,F]
        h = self.tcn(z_seq.transpose(1,2))    # [B,F',T]
        y = self.head(h)[...,-1]              # [B, H*n_out]
        return y.view(-1, self.horizons, self.n_out)  # [B,H,3]

class Phi2Flux(nn.Module):
    def __init__(self, horizons: List[int]):
        super().__init__()
        self.horizons = horizons
        self.enc = Spatial3DEncoder(c_in=2, c_mid=32, c_out=64)
        self.temporal = None  # lazy init

    def forward(self, x):  # x: [B,T,2,KX,KY,TH]
        B,T,C,KX,KY,TH = x.shape
        z = self.enc(x.view(B*T, C, KX, KY, TH))  # [B*T,F]
        F = z.shape[1]
        if (self.temporal is None) or (self.temporal.tcn[0].in_channels != F):
            self.temporal = TemporalTCN(f_in=F, f_hid=128, levels=4,
                                        horizons=len(self.horizons), n_out=3, dropout=0.1).to(z.device)
        z = z.view(B, T, -1)
        return self.temporal(z)

# ----------------------- Train / Eval / Logging -----------------------

def rmse(a,b,axis=0):
    return np.sqrt(np.mean((a-b)**2, axis=axis))

def train_one_epoch(model, loader, opt, device):
    model.train()
    crit = nn.SmoothL1Loss()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        yhat = model(xb)
        loss = crit(yhat, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    crit = nn.SmoothL1Loss(reduction='sum')
    total = 0.0
    preds, trues, bases = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yhat = model(xb)                       # [B,H,3]
        total += crit(yhat, yb.to(device)).item()
        preds.append(yhat.cpu().numpy())
        trues.append(yb.numpy())
        base = yb[:, :1, :].repeat(1, yb.shape[1], 1)  # [B,H,3] persistence
        bases.append(base.numpy())

    y_pred = np.concatenate(preds, 0)  # standardized
    y_true = np.concatenate(trues, 0)
    y_base = np.concatenate(bases, 0)

    y_pred_abs = y_pred * y_std + y_mean
    y_true_abs = y_true * y_std + y_mean
    y_base_abs = y_base * y_std + y_mean

    rmse_m = rmse(y_pred_abs, y_true_abs, axis=0)  # [H,3]
    rmse_b = rmse(y_base_abs, y_true_abs, axis=0)
    skill  = 1.0 - (rmse_m / (rmse_b + 1e-9))
    avg_loss = total/len(loader.dataset)
    return avg_loss, rmse_m, rmse_b, skill

def save_checkpoint(path, model, opt, epoch, extra=None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "extra": extra or {},
    }
    torch.save(state, path)

def plot_loss(log_path, out_png):
    epochs, tr, va = [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch'])); tr.append(float(row['train_loss'])); va.append(float(row['val_loss']))
    plt.figure()
    plt.plot(epochs, tr, label="train")
    plt.plot(epochs, va, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def bar_plot(matrix, labels_h, labels_c, title, out_png):
    # matrix: [H,3]
    H = matrix.shape[0]
    x = np.arange(H)
    plt.figure()
    for j in range(3):
        plt.bar(x + j*0.25, matrix[:, j], width=0.25, label=labels_c[j])
    plt.xticks(x + 0.25, labels_h)
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_png); plt.close()

def build_datasets(data_path, Tc, horizons,split=(0.6,0.2,0.2), seed=0):
    '''
    import numpy as np, torch
    d = np.load(data_path)
    phi = d["phi"]  # shape [T, Nr, Ntheta, Ntor, 2]
    flux = d["flux"]  # shape [T, 3]

    T = phi.shape[0]
    Hmax = max(horizons)
    starts = np.arange(T - Tc - Hmax + 1)
    np.random.seed(seed)
    np.random.shuffle(starts)

    n = len(starts)
    n_train = int(split[0]*n)
    n_val   = int(split[1]*n)
    idx_train = starts[:n_train]
    idx_val   = starts[n_train:n_train+n_val]
    idx_test  = starts[n_train+n_val:]

    def make_split(indices):
        X, Y = [], []
        for s in indices:
            x = phi[s:s+Tc, ...]
            x = np.transpose(x, (0,4,1,2,3))  # [Tc, 2, Nr, Ntheta, Ntor]
            y = np.stack([flux[s+Tc+h-1] for h in horizons], axis=0)
            X.append(x)
            Y.append(y)
        return np.stack(X), np.stack(Y)

    Xtr, Ytr = make_split(idx_train)
    Xva, Yva = make_split(idx_val)
    Xte, Yte = make_split(idx_test)

    # normalize Y
    y_mean = Ytr.mean(axis=(0,1), keepdims=True)
    y_std  = Ytr.std(axis=(0,1), keepdims=True) + 1e-8
    Ytr = (Ytr - y_mean)/y_std
    Yva = (Yva - y_mean)/y_std
    Yte = (Yte - y_mean)/y_std

    train_ds = PhiFluxDataset(Xtr, Ytr, y_mean, y_std)
    val_ds   = PhiFluxDataset(Xva, Yva, y_mean, y_std)
    test_ds  = PhiFluxDataset(Xte, Yte, y_mean, y_std)
    '''

    d = np.load(data_path)
    phi = d['phi']
    assert 'flux' in d, "NPZ must include integrated flux [T,3]"
    flux = d['flux']

    T = phi.shape[0]
    t_train_end = int(split[0]*T)
    t_val_end   = int((split[0]+split[1])*T)
    spec = WindowSpec(Tc, horizons, 1)
    ds_train = PhiFluxDataset(phi, flux, spec, (0, t_train_end-Tc), standardize=True)
    ds_val   = PhiFluxDataset(phi, flux, spec, (t_train_end, t_val_end-Tc), standardize=True)
    ds_test  = PhiFluxDataset(phi, flux, spec, (t_val_end, Tc-1), standardize=True)


    return ds_train, ds_val, ds_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--Tc', type=int, default=64)
    ap.add_argument('--horizons', type=int, nargs='+', default=[1,2,5])
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--log_dir', type=str, default='./runs')
    ap.add_argument('--save_ckpt', type=int, default=1)
    ap.add_argument('--ckpt_every', type=int, default=5)
    ap.add_argument('--plots', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_csv = os.path.join(args.log_dir, "metrics.csv")
    best_ckpt = os.path.join(args.log_dir, "best.pt")
    last_ckpt = os.path.join(args.log_dir, "last.pt")

    d = np.load(args.data)
    phi = d['phi']
    assert 'flux' in d, "NPZ must include integrated flux [T,3]"
    flux = d['flux']

    T = phi.shape[0]
    t_train_end = int(0.6*T)
    t_val_end   = int(0.8*T)
    spec = WindowSpec(Tc=args.Tc, horizons=args.horizons, stride=args.stride)
    ds_train = PhiFluxDataset(phi, flux, spec, (0, t_train_end-args.Tc), standardize=True)
    ds_val   = PhiFluxDataset(phi, flux, spec, (t_train_end, t_val_end-args.Tc), standardize=True)
    ds_test  = PhiFluxDataset(phi, flux, spec, (t_val_end, T-args.Tc-1), standardize=True)

# ==========================================================
# 🔍 Dataset diagnostics: check target scaling (Qi, Qe, Γ)
# ==========================================================
    def describe_flux(name, ds):
       y_mean = getattr(ds, "y_mean", None)
       y_std = getattr(ds, "y_std", None)
       print(f"\n[{name}] dataset:")
       if y_mean is not None and y_std is not None:
           print(f"  y_mean: {np.round(y_mean, 5)}")
           print(f"  y_std : {np.round(y_std, 5)}")
       else:
           print("  (no stored normalization stats)")

       # sample actual y values from first 100 windows
       n = min(len(ds), 100)
       Y = []
       for i in range(n):
           _, y = ds[i]
           Y.append(y)
       Y = np.stack(Y)
       print(f"  sample Y: min={Y.min():.3e}, max={Y.max():.3e}, mean={Y.mean():.3e}, std={Y.std():.3e}")

        # optional quick plot
       plt.figure(figsize=(6,3))
       plt.hist(Y[...,0].ravel(), bins=50, alpha=0.5, label="Qi")
       plt.hist(Y[...,1].ravel(), bins=50, alpha=0.5, label="Qe")
       plt.hist(Y[...,2].ravel(), bins=50, alpha=0.5, label="Gamma")
       plt.title(f"{name} flux distributions (first 100 samples)")
       plt.legend()
       plt.tight_layout()
       plt.savefig(f"{args.log_dir}/{name}_flux_hist.png", dpi=160)
       plt.close()

    describe_flux("train", ds_train)
    describe_flux("val",   ds_val)
    describe_flux("test",  ds_test)
    print("✅ Saved flux histograms in", args.log_dir)
# ==========================================================






    tr_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True)
    va_loader = DataLoader(ds_val,   batch_size=args.batch, shuffle=False)
    te_loader = DataLoader(ds_test,  batch_size=args.batch, shuffle=False)

    device = torch.device(args.device)
    model = Phi2Flux(horizons=args.horizons).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.SmoothL1Loss()

    # CSV logger
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss"])

    best_val = float('inf')
   
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, tr_loader, opt, device)
        val_loss, _, _, _ = evaluate(model, va_loader, device, ds_val.y_mean[0], ds_val.y_std[0])
        dt = time.time()-t0
        print(f"[{ep:02d}] train={tr_loss:.4f}  val={val_loss:.4f}  ({dt:.1f}s)")
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f); writer.writerow([ep, tr_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            if args.save_ckpt:
                save_checkpoint(best_ckpt, model, opt, ep, extra={"best_val": best_val})

        if args.save_ckpt and (ep % args.ckpt_every == 0):
            save_checkpoint(os.path.join(args.log_dir, f"epoch_{ep}.pt"), model, opt, ep)

    # End of training: save last
    if args.save_ckpt:
        save_checkpoint(last_ckpt, model, opt, args.epochs)

    # Final evaluation on test split
    test_loss, rmse_m, rmse_b, skill = evaluate(model, te_loader, device, ds_test.y_mean[0], ds_test.y_std[0])
    print("=== Test Summary ===")
    for i,h in enumerate(args.horizons):
        print(f"Δ={h:>2d}: RMSE(model)=[Qi {rmse_m[i,0]:.4f}, Qe {rmse_m[i,1]:.4f}, Γ {rmse_m[i,2]:.4f}]  "
              f"RMSE(persist)=[{rmse_b[i,0]:.4f}, {rmse_b[i,1]:.4f}, {rmse_b[i,2]:.4f}]  "
              f"Skill=[{skill[i,0]:.3f}, {skill[i,1]:.3f}, {skill[i,2]:.3f}]")

    # Plots
    if args.plots:
        plot_loss(log_csv, os.path.join(args.log_dir, "loss_curve.png"))
        labels_h = [f"Δ{h}" for h in args.horizons]
        labels_c = ["Qi","Qe","Γ"]
        bar_plot(rmse_m, labels_h, labels_c, "RMSE (Model)", os.path.join(args.log_dir, "rmse_bar.png"))
        bar_plot(skill,  labels_h, labels_c, "Skill vs Persistence", os.path.join(args.log_dir, "skill_bar.png"))

    # JSON summary
    summary = {
        "best_val_loss": best_val,
        "test_loss": float(test_loss),
        "rmse_model": rmse_m.tolist(),
        "rmse_persistence": rmse_b.tolist(),
        "skill": skill.tolist(),
        "log_dir": args.log_dir,
        "checkpoints": {
            "best": best_ckpt if args.save_ckpt else None,
            "last": last_ckpt if args.save_ckpt else None
        }
    }
    with open(os.path.join(args.log_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
