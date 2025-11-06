#After you run something that writes predictions, e.g.:

#python resume_or_infer_deep.py \
#    --data ./mnt/data/myrun_phi_flux.npz \
#    --nn_path ./mnt/data/bin.cgyro.nn \
#    --Tc 32 \
#    --horizons 1 5 10 \
#    --out_npz ./mnt/data/myrun_preds_horizons.npz \
#    --device cuda


#Then call:

#python analyze_horizons.py --npz ./mnt/data/myrun_preds_horizons.npz --phys_units 1



#!/usr/bin/env python
import argparse
import numpy as np

def compute_stats(y_true, y_pred):
    """
    y_true, y_pred: arrays of shape [N, 3]  (Qi, Qe, Gamma)
    Returns:
      rmse: [3]
      bias: [3]  (mean(pred - true))
      mae : [3]
    """
    diff = y_pred - y_true
    mse  = (diff ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    bias = diff.mean(axis=0)
    mae  = np.abs(diff).mean(axis=0)
    return rmse, bias, mae

def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-horizon metrics from NPZ with y_true, y_pred, horizons."
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to NPZ file containing y_true, y_pred, horizons"
    )
    parser.add_argument(
        "--phys_units",
        type=int,
        default=1,
        help="1 if y_true/y_pred are in physical units already, 0 if normalized (just label the printout)."
    )
    args = parser.parse_args()

    d = np.load(args.npz)
    if not all(k in d for k in ["y_true", "y_pred", "horizons"]):
        raise KeyError("NPZ must contain 'y_true', 'y_pred', and 'horizons' arrays.")

    y_true = d["y_true"]     # [N, H, 3]
    y_pred = d["y_pred"]     # [N, H, 3]
    horizons = d["horizons"] # [H]

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    N, H, C = y_true.shape
    if C != 3:
        print(f"Warning: expected 3 channels (Qi,Qe,Gamma), got C={C}")
    print(f"Loaded: N={N} samples, H={H} horizons, C={C} channels.")

    unit_str = "physical" if args.phys_units else "normalized (z-score)"
    print(f"Assuming targets are in {unit_str} units.\n")

    labels = ["Qi", "Qe", "Gamma"]

    # ----- Global metrics over all horizons -----
    y_true_flat = y_true.reshape(-1, C)
    y_pred_flat = y_pred.reshape(-1, C)
    rmse_all, bias_all, mae_all = compute_stats(y_true_flat, y_pred_flat)

    print("=== Global metrics over all horizons ===")
    for i, lbl in enumerate(labels[:C]):
        print(f"  {lbl}: RMSE={rmse_all[i]:.4f}, MAE={mae_all[i]:.4f}, Bias={bias_all[i]:+.4f}")
    print()

    # ----- Per-horizon metrics -----
    print("=== Per-horizon metrics ===")
    header = f"{'H':>6} | {'RMSE_Qi':>9} {'RMSE_Qe':>9} {'RMSE_Gam':>9} | {'Bias_Qi':>9} {'Bias_Qe':>9} {'Bias_Gam':>9}"
    print(header)
    print("-" * len(header))

    for hi, h in enumerate(horizons):
        yt = y_true[:, hi, :]  # [N, C]
        yp = y_pred[:, hi, :]  # [N, C]
        rmse, bias, mae = compute_stats(yt, yp)
        # pad if C < 3
        rm = list(rmse) + [np.nan]*(3 - len(rmse))
        bs = list(bias) + [np.nan]*(3 - len(bias))
        print(f"{int(h):6d} | "
              f"{rm[0]:9.4f} {rm[1]:9.4f} {rm[2]:9.4f} | "
              f"{bs[0]:+9.4f} {bs[1]:+9.4f} {bs[2]:+9.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

