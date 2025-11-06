#!/usr/bin/env python3
import argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi_npz', required=True)
    ap.add_argument('--ky_npz',  required=True)
    ap.add_argument('--t_init',type=int, default=0 )
    ap.add_argument('--out',     required=True)
    args = ap.parse_args()

    dphi = np.load(args.phi_npz)
    dky  = np.load(args.ky_npz)
    T_in =args.t_init
    # Expect: dphi['phi'] -> [T, KX, KY, TH, 2]
    T_phi = dphi['phi'].shape[0]

    # From your ky parser (time-major):
    #   Qi, Qe, Gamma → [T_ky]
    Qi = dky['Qi']; Qe = dky['Qe']; Gamma = dky['Gamma']
    T_ky = Qi.shape[0]
    T = min(T_phi+T_in , T_ky) # cropped off T_in when reading T_in, putting it  back here 
    if T_phi != T_ky:
        print(f"[merge] Warning: phi T={T_phi}, ky T={T_ky} -> truncating to T={T}")

    flux = np.stack([Qi[T_in:T], Qe[T_in:T], Gamma[T_in:T]], axis=1).astype(np.float32)  # [T,3]

    # Copy all phi arrays, but truncate time to T to keep consistency
    out = {k: v[0:T] if (isinstance(v, np.ndarray) and v.shape[0]==T_phi) else v
           for k,v in dphi.items()}
    out['flux'] = flux

    # Optional: keep ky-resolved spectra for later analysis if present
    for k in ['Qi_ky','Qe_ky','Gamma_ky_total','Gamma_i','Gamma_e','ky','dky','nt','ntime']:
        if k in dky: out[k] = dky[k]

    np.savez_compressed(args.out, **out)
    print(f"[merge] wrote {args.out} with flux shape {flux.shape} and phi time T={T-T_in}")

if __name__ == '__main__':
    main()

