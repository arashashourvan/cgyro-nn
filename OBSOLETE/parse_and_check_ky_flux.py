#!/usr/bin/env python3
"""
parse_and_check_kyflux.py

Stand-alone checker & extractor for CGYRO bin.cgyro.ky_flux

Layout (matches cgyro/data.py):
    np.fromfile(..., dtype=np.float32)  --->  reshape(order='F') to:
    (n_species=2, m=3, n_field=2, n_n, n_time)

m index:
  0 -> Gamma (particle flux)
  1 -> Q     (energy flux)
  2 -> momentum (not used here)

field:
  0 -> phi (electrostatic)
  1 -> A_parallel

species:
  0 -> ions
  1 -> electrons

Outputs:
  - Quick dimension check report
  - Optional .npz with time series (Qi(t), Qe(t), Gamma_total(t))
  - Optional ky-resolved arrays (Qi_ky, Qe_ky, Gamma_i_ky, Gamma_e_ky)

Usage:
  python parse_and_check_kyflux.py \
    --bin ./bin.cgyro.ky_flux \
    --n_n 16 --n_time 856 \
    --dky 0.067 --k0 0.0 \
    --save_npz ./kyflux_checked.npz
"""

import argparse, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to bin.cgyro.ky_flux")
    ap.add_argument("--n_n", type=int, required=True, help="ky/binormal count (often 16 or 15)")
    ap.add_argument("--n_time", type=int, required=True, help="number of time slices")
    ap.add_argument("--dky", type=float, default=None, help="Δky for integration (optional)")
    ap.add_argument("--k0", type=float, default=0.0, help="ky offset (optional)")
    ap.add_argument("--save_npz", type=str, default=None, help="Write parsed arrays to this .npz")
    ap.add_argument("--keep_ky", action="store_true", help="Also store ky-resolved spectra in NPZ")
    args = ap.parse_args()

    path = args.bin
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ns, nm, nf = 2, 3, 2   # fixed by CGYRO
    base = ns * nf * args.n_n * args.n_time

    # --- Read raw floats (CGYRO uses float32) ---
    data = np.fromfile(path, dtype=np.float32)
    total = data.size

    print(f"\nFile: {path}")
    print(f"Total float32 count: {total:,}")
    print(f"Expected per m-slice block (2*2*n_n*n_time): {base:,}")

    if total % base != 0:
        raise ValueError(f"Float count {total} is not divisible by 2*2*n_n*n_time = {base}. "
                         f"Check n_n / n_time or file path.")

    m = total // base
    print(f"Deduced m = {m}  (should be 3: Gamma, Q, momentum)")
    if m != nm:
        print("WARNING: m != 3. Proceeding, but confirm your n_n and n_time.")

    # --- Reshape with Fortran order (matches data.py) ---
    arr = np.reshape(data[:base*m], (ns, m, nf, args.n_n, args.n_time), order='F')

    print("\nReshaped ky_flux dims (species, m, field, n_n, n_time):", arr.shape)
    print(" Index meanings: species [0=ions, 1=electrons], m [0=Gamma,1=Q,2=momentum], field [0=phi,1=A_par]")

    # --- Extract φ-only fluxes ---
    fld_phi = 0
    sp_i, sp_e = 0, 1
    m_gamma, m_Q = 0, 1

    Gamma_i_ky = arr[sp_i, m_gamma, fld_phi]  # [n_n, n_time]
    Gamma_e_ky = arr[sp_e, m_gamma, fld_phi]
    Qi_ky      = arr[sp_i, m_Q,     fld_phi]
    Qe_ky      = arr[sp_e, m_Q,     fld_phi]

    print("\nQuick stats (φ-only, over ky & time):")
    def stats(name, a):
        a = np.asarray(a)
        print(f"  {name:12s}: shape={a.shape}  min={a.min():.3e}  max={a.max():.3e}  mean={a.mean():.3e}")
    stats("Gamma_i_ky", Gamma_i_ky)
    stats("Gamma_e_ky", Gamma_e_ky)
    stats("Qi_ky",      Qi_ky)
    stats("Qe_ky",      Qe_ky)

    # --- Optional: integrate over ky (time series) ---
    out = {"shape": np.array(arr.shape, dtype=np.int32)}
    if args.dky is not None:
        dky = float(args.dky)
        ky = args.k0 + np.arange(args.n_n, dtype=np.float32)*dky

        # integrate over ky axis=0 → [n_time]
        Gamma_i = (Gamma_i_ky * dky).sum(axis=0)
        Gamma_e = (Gamma_e_ky * dky).sum(axis=0)
        Qi      = (Qi_ky      * dky).sum(axis=0)
        Qe      = (Qe_ky      * dky).sum(axis=0)
        Gamma   = Gamma_i + Gamma_e

        print("\nIntegrated over ky (using Δky):")
        def stats_t(name, a):
            a = np.asarray(a)
            print(f"  {name:12s}: shape={a.shape}  min={a.min():.3e}  max={a.max():.3e}  mean={a.mean():.3e}")
        stats_t("Gamma_i(t)", Gamma_i)
        stats_t("Gamma_e(t)", Gamma_e)
        stats_t("Gamma(t)",   Gamma)
        stats_t("Qi(t)",      Qi)
        stats_t("Qe(t)",      Qe)

        out.update({
            "ky": ky, "dky": np.float32(dky),
            "Gamma_i": Gamma_i.astype(np.float32),
            "Gamma_e": Gamma_e.astype(np.float32),
            "Gamma":   Gamma.astype(np.float32),
            "Qi":      Qi.astype(np.float32),
            "Qe":      Qe.astype(np.float32),
        })

    # --- Optional: save NPZ ---
    if args.save_npz:
        if args.keep_ky:
            out.update({
                "Gamma_i_ky": Gamma_i_ky.astype(np.float32),   # [n_n, n_time]
                "Gamma_e_ky": Gamma_e_ky.astype(np.float32),
                "Qi_ky":      Qi_ky.astype(np.float32),
                "Qe_ky":      Qe_ky.astype(np.float32),
            })
        np.savez_compressed(args.save_npz, **out)
        print(f"\nSaved: {args.save_npz}")

if __name__ == "__main__":
    main()

