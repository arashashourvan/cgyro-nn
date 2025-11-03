#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from phi2flux_deep import Phi2FluxDeep

# -------------------------------
# CONFIGURATION
# -------------------------------
BEST_PT = "./mnt/data/myrun_logs_deep/best.pt"     # or your .nn file
NPZ_PATH = "./mnt/data/myrun_phi_flux.npz"
DEVICE = "cpu"   # use "cuda" if GPU available
BATCH_SIZE = 8
HORIZONS = [1, 5, 10]   # which horizons to test

# -------------------------------
# LOAD CHECKPOINT
# -------------------------------
print(f"Loading checkpoint: {BEST_PT}")
ckpt = torch.load(BEST_PT, map_location=DEVICE)
cfg = ckpt.get("cfg", {})
state = ckpt["model_state_dict"]

# Rebuild model
model = Phi2FluxDeep(
    Tc=cfg.get("Tc", 64),
    horizons=cfg.get("horizons", HORIZONS),
    base_channels=cfg.get("base_channels", 32),
    depth=cfg.get("depth", 8),
    tcn_channels=cfg.get("tcn_channels", 128),
    tcn_blocks=cfg.get("tcn_blocks", 3),
    dropout=cfg.get("dropout", 0.2),
    norm=cfg.get("norm", "bn")
).to(DEVICE)


model.load_state_dict(state, strict=False)
model.eval()
print("✅ Model loaded successfully.\n")

# -------------------------------
# LOAD DATA
# -------------------------------
print(f"Loading dataset: {NPZ_PATH}")
data = np.load(NPZ_PATH)
X = data["phi"]  # shape: [Nt, Nr, Nθ, Ntor, 2]
Y = data["flux"] # shape: [Nt, 3]  -> Qi, Qe, Γ

Nt, Nr, Ntheta, Ntor, _ = X.shape
#N_samples = Nt - Tc - max(HORIZONS)

# Prepare rolling input windows

# -------------------------------
# BUILD WINDOWS (ensure channels-first per window)
# -------------------------------
Tc = cfg.get("Tc", 64)
HORIZONS = cfg.get("horizons", HORIZONS)   # keep your chosen test horizons
Hmax = max(HORIZONS)
N_samples = Nt - Tc - Hmax + 1
if N_samples <= 0:
    raise ValueError(f"Not enough time points: Nt={Nt}, Tc={Tc}, Hmax={Hmax}")

X_seq, Y_true = [], []
for s in range(N_samples):
    # X[s:s+Tc] is [Tc, R, TH, TOR, 2] from the NPZ
    xw = X[s:s+Tc]                            # [Tc, R, TH, TOR, 2]
    xw = np.transpose(xw, (0, 4, 1, 2, 3))    # --> [Tc, 2, R, TH, TOR]
    # Use the largest horizon's target index; (you can also stack multiple horizons here if desired)
    # We'll predict all horizons via the model heads; for plotting single-step curves we pick one.
    # If you want per-horizon true targets later, you can build a [H,3] stack similarly.
    Y_true.append(Y[s+Tc+Hmax-1])             # [3]
    X_seq.append(xw)

X_seq = np.asarray(X_seq, dtype=np.float32)   # [N, Tc, 2, R, TH, TOR]
Y_true = np.asarray(Y_true, dtype=np.float32) # [N, 3]
print(f"Prepared {len(X_seq)} samples → X: {X_seq.shape}, Y: {Y_true.shape}")


# -------------------------------
# RUN INFERENCE
# -------------------------------
preds = []
with torch.no_grad():
    for i in range(0, len(X_seq), BATCH_SIZE):
        #xb = torch.tensor(X_seq[i:i+BATCH_SIZE], dtype=torch.float32, device=DEVICE)
        xb = torch.from_numpy(X_seq[i:i+BATCH_SIZE]).to(device=DEVICE, dtype=torch.float32)
        out = model(xb)
        preds.append(out.cpu().numpy())
preds = np.concatenate(preds, axis=0)
print(f"Predictions shape: {preds.shape}")  # [N, H, 3]

# -------------------------------
# Build ground truth for ALL horizons → [N, H, 3]
# -------------------------------
HORIZONS = cfg.get("horizons", HORIZONS)  # keep your chosen test horizons
Hmax = max(HORIZONS)
Tc = cfg.get("Tc", 64)

# Number of windows must match how you built X_seq (use +1)
N_samples = Nt - Tc - Hmax + 1
if N_samples != preds.shape[0]:
    print(f"[warn] N_samples({N_samples}) != preds.shape[0]({preds.shape[0]}) — adjusting to preds N")
    N_samples = preds.shape[0]

Y_all = np.empty((N_samples, len(HORIZONS), 3), dtype=np.float32)
for i, h in enumerate(HORIZONS):
    # align with window start s → target index is s + Tc + h - 1
    for s in range(N_samples):
        Y_all[s, i, :] = Y[s + Tc + h - 1]

print(f"Ground-truth shape: {Y_all.shape}")  # [N, H, 3]

# -------------------------------
# Choose a horizon to visualize (e.g., the largest)
# -------------------------------
h_sel = len(HORIZONS) - 1          # last horizon in your list
h_val = HORIZONS[h_sel]
Qi_true = Y_all[:, h_sel, 0]
Qe_true = Y_all[:, h_sel, 1]
G_true  = Y_all[:, h_sel, 2]

Qi_pred = preds[:, h_sel, 0]
Qe_pred = preds[:, h_sel, 1]
G_pred  = preds[:, h_sel, 2]

print(f"Using horizon h={h_val} → arrays: true {Qi_true.shape}, pred {Qi_pred.shape}")

# -------------------------------
# Plots (now sizes match)
# -------------------------------
def scatter_plot(y_true, y_pred, label, color):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5, color=color)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(f"{label} @ horizon {h_val}")
    plt.tight_layout()
    plt.savefig(f"./mnt/data/{label}_scatter_h{h_val}.png", dpi=160)
    plt.close()

scatter_plot(Qi_true, Qi_pred, "Qi", "tab:red")
scatter_plot(Qe_true, Qe_pred, "Qe", "tab:blue")
scatter_plot(G_true,  G_pred,  "Gamma", "tab:green")

# RMSE at selected horizon
rmse = np.sqrt(np.mean((np.stack([Qi_pred, Qe_pred, G_pred], axis=1)
                        - np.stack([Qi_true, Qe_true, G_true], axis=1))**2, axis=0))
print(f"RMSE @ h={h_val}: Qi={rmse[0]:.3f}, Qe={rmse[1]:.3f}, Γ={rmse[2]:.3f}")

# Time-series overlay at selected horizon (first 500 samples for readability)
import matplotlib.pyplot as plt
K = min(500, Qi_true.shape[0])
plt.figure(figsize=(6,3))
plt.plot(Qi_true[:K], label=f"Qi true h={h_val}")
plt.plot(Qi_pred[:K], "--", label=f"Qi pred h={h_val}")
plt.plot(Qe_true[:K], label=f"Qe true h={h_val}")
plt.plot(Qe_pred[:K], "--", label=f"Qe pred h={h_val}")
plt.plot(G_true[:K],  label=f"Γ true h={h_val}")
plt.plot(G_pred[:K],  "--", label=f"Γ pred h={h_val}")
plt.legend(ncol=3, fontsize=8)
plt.xlabel("Sample index"); plt.ylabel("Flux (normalized)")
plt.title(f"Predicted vs True @ horizon {h_val}")
plt.tight_layout()
plt.savefig(f"./mnt/data/flux_time_series_h{h_val}.png", dpi=160)
plt.close()
