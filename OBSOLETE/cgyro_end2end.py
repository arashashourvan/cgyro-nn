#!/usr/bin/env python3
"""
cgyro_end2end.py
----------------
One-shot pipeline for:
  (1) Parsing CGYRO φ binary  -> NPZ
  (2) Parsing ky-flux binary  -> integrate to totals and merge into NPZ
  (3) Training a 3-output model (Qi, Qe, Gamma)

Requirements: the sibling scripts must be available on disk:
  - parse_cgyro_phi.py
  - parse_cgyro_kyflux.py
  - phi2flux_3species.py

Example (matches your uploaded sample files):
  python cgyro_end2end.py \
    --phi_bin /mnt/data/bin.cgyro.kxky_phi --Nr 324 --Ntheta 1 --Ntor 16 --force_raw 1 \
    --ky_bin  /mnt/data/bin.cgyro.ky_flux  --dky 0.067 \
    --Tc 64 --horizons 1 2 5 --epochs 20 \
    --out_root /mnt/data/myrun

This will produce:
  - /mnt/data/myrun_phi.npz       (phi only)
  - /mnt/data/myrun_phi_flux.npz  (phi + ky_flux + ky + flux totals)
  - a training run with stats printed to stdout
"""
import argparse, os, sys, subprocess, shlex, json
from pathlib import Path

def run(cmd):
    print(">>", cmd)
    res = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def main():
    ap = argparse.ArgumentParser()
    # φ parsing
    ap.add_argument('--phi_bin', required=True, help='Path to CGYRO φ binary')
    ap.add_argument('--Nr', type=int, required=True)
    ap.add_argument('--Ntheta', type=int, required=True)
    ap.add_argument('--Ntor', type=int, required=True)
    ap.add_argument('--force_raw', type=int, default=0)
    ap.add_argument('--force_fortran', type=int, default=0)
    ap.add_argument('--stride_t', type=int, default=1)
    ap.add_argument('--ds_r', type=int, default=1)
    ap.add_argument('--ds_theta', type=int, default=1)
    ap.add_argument('--ds_tor', type=int, default=1)
    ap.add_argument('--fft_theta', type=int, default=0)
    ap.add_argument('--fft_tor', type=int, default=0)
    ap.add_argument('--transport', type=str, default='')
    # ky-flux parsing
    ap.add_argument('--ky_bin', required=True, help='Path to ky-resolved flux binary')
    ap.add_argument('--dky', type=float, required=True, help='ky spacing for integration')
    ap.add_argument('--k0', type=float, default=0.0)
    # Training
    ap.add_argument('--Tc', type=int, default=64)
    ap.add_argument('--horizons', type=int, nargs='+', default=[1,2,5])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='auto')
    # Output root
    ap.add_argument('--out_root', required=True, help='Prefix for outputs')
    args = ap.parse_args()

    root = Path(args.out_root).expanduser()
    root.parent.mkdir(parents=True, exist_ok=True)

    # Paths to helper scripts (assumed to be alongside this script or under /mnt/data)
    here = Path(__file__).resolve().parent
    parse_phi = str(here / "parse_cgyro_phi.py")
    parse_ky  = str(here / "parse_cgyro_kyflux.py")
    train_py  = str(here / "phi2flux_3species.py")
    for p in (parse_phi, parse_ky, train_py):
        if not Path(p).exists():
            # also try /mnt/data fallback
            alt = Path("/mnt/data") / Path(p).name
            if alt.exists():
                p = str(alt)

    # 1) Parse φ
    phi_npz = f"{args.out_root}_phi.npz"
    cmd_phi = f"python {parse_phi} --bin {shlex.quote(args.phi_bin)} " \
              f"--Nr {args.Nr} --Ntheta {args.Ntheta} --Ntor {args.Ntor} " \
              f"--stride_t {args.stride_t} --ds_r {args.ds_r} --ds_theta {args.ds_theta} --ds_tor {args.ds_tor} " \
              f"--fft_theta {args.fft_theta} --fft_tor {args.fft_tor} " \
              f"{'--force_raw 1' if args.force_raw else ''} {'--force_fortran 1' if args.force_fortran else ''} " \
              f"{f'--transport {shlex.quote(args.transport)}' if args.transport else ''} " \
              f"--out {shlex.quote(phi_npz)}"
    run(cmd_phi)

    # 2) Parse ky-flux and merge into phi NPZ (overwrites/creates 'flux' totals)
    phi_flux_npz = f"{args.out_root}_phi_flux.npz"
    # Copy phi npz to phi_flux npz path
    import shutil
    shutil.copyfile(phi_npz, phi_flux_npz)
    cmd_ky = f"python {parse_ky} --bin {shlex.quote(args.ky_bin)} " \
             f"--phi_npz {shlex.quote(phi_flux_npz)} --dky {args.dky} --k0 {args.k0} " \
             f"--merge_into {shlex.quote(phi_flux_npz)}"
    run(cmd_ky)

    # 3) Train model
    device = args.device if args.device != 'auto' else ''
    dev_flag = f"--device {device}" if device else ""
    horizons = " ".join(str(h) for h in args.horizons)
    cmd_train = f"python {train_py} --data {shlex.quote(phi_flux_npz)} " \
                f"--Tc {args.Tc} --horizons {horizons} --epochs {args.epochs} " \
                f"--batch {args.batch} --lr {args.lr} {dev_flag}"
    run(cmd_train)

    print("Pipeline complete. Outputs:")
    print(f"  φ only NPZ:     {phi_npz}")
    print(f"  φ+flux NPZ:     {phi_flux_npz}")

if __name__ == "__main__":
    main()
