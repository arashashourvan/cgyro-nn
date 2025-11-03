#!/usr/bin/env python3
import argparse, os, torch
from phi2flux_deep import Phi2FluxDeep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--nn_path", required=True)
    ap.add_argument("--Tc", type=int, required=True)
    # Optional fallbacks if cfg isn’t in best.pt
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--tcn_channels", type=int, default=128)
    ap.add_argument("--tcn_blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--norm", default="bn")
    args = ap.parse_args()

    best_path = os.path.join(args.log_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No checkpoint at {best_path}")

    ckpt = torch.load(best_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    state = ckpt["model_state_dict"]

    Tc = cfg.get("Tc", args.Tc)
    horizons = cfg.get("horizons", [1, 5, 10])

    model = Phi2FluxDeep(
        Tc=Tc,
        horizons=horizons,
        base_channels=cfg.get("base_channels", args.base_channels),
        depth=cfg.get("depth", args.depth),
        tcn_channels=cfg.get("tcn_channels", args.tcn_channels),
        tcn_blocks=cfg.get("tcn_blocks", args.tcn_blocks),
        dropout=cfg.get("dropout", args.dropout),
        norm=cfg.get("norm", args.norm),
    )
    model.load_state_dict(state, strict=False)

    payload = {
        "state_dict": model.state_dict(),
        "arch": {
            "Tc": Tc,
            "base_channels": args.base_channels,
            "depth": args.depth,
            "tcn_channels": args.tcn_channels,
            "tcn_blocks": args.tcn_blocks,
            "dropout": args.dropout,
            "norm": args.norm,
        },
        "train_horizons": horizons,
        "y_mean": None,
        "y_std": None,
        "meta": {
            "source": best_path,
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
        },
    }
    os.makedirs(os.path.dirname(args.nn_path), exist_ok=True)
    torch.save(payload, args.nn_path)
    print(f"[save_nn] wrote {args.nn_path}")

if __name__ == "__main__":
    main()
