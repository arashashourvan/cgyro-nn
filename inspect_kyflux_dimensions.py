#!/usr/bin/env python3
"""
inspect_kyflux_dimensions.py
----------------------------
Diagnose CGYRO ky_flux Fortran-unformatted binary.

Assumed per-time record layout (float32 unless --dtype=float64):
    [n_species=2, m=3, n_field=2, n_r=NR, n_t=NT]

m index:
  0 -> Gamma (particle flux), 1 -> Q (energy flux), 2 -> momentum (ignored)

n_field:
  0 -> phi (electrostatic), 1 -> A_parallel

species:
  0 -> ions, 1 -> electrons

What it does:
  - Autodetects 4B vs 8B record markers.
  - Skips zero-length/garbage headers at start.
  - Verifies each record has the expected float count.
  - Reports number of time slices T.
  - Extracts φ-only Qi, Qe, Gamma spectra for the *first* record to print quick stats.
  - Optional: --stop_after to stop early (fast check).

Usage:
  python inspect_kyflux_dimensions.py \
      --bin ./mnt/data/bin.cgyro.ky_flux \
      --nr 324 --nt 16 \
      --dtype float32 \
      --stop_after 5
"""

import argparse, os, struct
import numpy as np

def read_fortran_record(f, dtype=np.float32):
    """
    Read one Fortran unformatted record.
    Auto-detect 4-byte or 8-byte record markers.
    Returns np.ndarray of the record payload (dtype), or None at EOF.
    Skips invalid/empty records gracefully (caller can decide to continue).
    """
    pos = f.tell()

    # Try 4-byte header
    h4 = f.read(4)
    if not h4:
        return None  # EOF
    n4 = struct.unpack('i', h4)[0]

    if 0 < n4 < 10**9:
        buf = f.read(n4)
        if len(buf) != n4:
            return None
        t4 = f.read(4)
        if len(t4) != 4:
            return None
        return np.frombuffer(buf, dtype=dtype)

    # Try 8-byte header
    f.seek(pos)
    h8 = f.read(8)
    if not h8:
        return None
    n8 = struct.unpack('q', h8)[0]

    if 0 < n8 < 10**12:
        buf = f.read(n8)
        if len(buf) != n8:
            return None
        t8 = f.read(8)
        if len(t8) != 8:
            return None
        return np.frombuffer(buf, dtype=dtype)

    # Neither worked: treat as invalid (caller may choose to stop)
    return np.array([], dtype=dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True, help='Path to bin.cgyro.ky_flux')
    ap.add_argument('--nr', type=int, default=324, help='n_r (radial index count)')
    ap.add_argument('--nt', type=int, default=16, help='n_t (binormal ky count)')
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32','float64'])
    ap.add_argument('--stop_after', type=int, default=0, help='Stop after reading N records (0 = read all)')
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == 'float32' else np.float64
    per_time_expected = 2 * 3 * 2 * args.nr * args.nt  # n_species * m * n_field * nr * nt

    sz = os.stat(args.bin).st_size
    print(f"File: {args.bin}")
    print(f"Total size: {sz:,} bytes")
    print(f"Expect per record floats: {per_time_expected} = 2*3*2*{args.nr}*{args.nt} ({args.dtype})")

    rec_count = 0
    bad_sizes = 0
    first_good = None

    with open(args.bin, 'rb') as f:
        while True:
            rec = read_fortran_record(f, dtype=dtype)
            if rec is None:
                break  # EOF
            if rec.size == 0:
                # Skip empty/garbage records
                continue

            rec_count += 1
            if rec.size != per_time_expected:
                bad_sizes += 1
                print(f"  ! Record {rec_count}: size {rec.size} floats (expected {per_time_expected})")
            else:
                if first_good is None:
                    first_good = rec.copy()

            if args.stop_after and rec_count >= args.stop_after:
                print(f"Stopping early after {rec_count} records per --stop_after.")
                break

    print(f"\nDetected records (time slices): T = {rec_count}")
    if bad_sizes > 0:
        print(f"WARNING: {bad_sizes} record(s) had unexpected size.")

    if first_good is not None:
        # Reshape and print quick stats for φ-only Qi/Qe/Γ from the *first* valid record
        try:
            rec_5d = first_good.reshape(2, 3, 2, args.nr, args.nt)  # [species, m, field, nr, nt]
        except Exception as e:
            print(f"Could not reshape first record to [2,3,2,{args.nr},{args.nt}]: {e}")
            return

        # indices
        sp_i, sp_e = 0, 1
        m_gamma, m_Q = 0, 1
        field_phi = 0

        Gamma_i_ky = rec_5d[sp_i, m_gamma, field_phi]   # [nr, nt]
        Gamma_e_ky = rec_5d[sp_e, m_gamma, field_phi]
        Qi_ky      = rec_5d[sp_i, m_Q,     field_phi]
        Qe_ky      = rec_5d[sp_e, m_Q,     field_phi]

        def stats(name, a):
            a = np.asarray(a)
            print(f"  {name}: shape {a.shape}  min={a.min():.3e}  max={a.max():.3e}  mean={a.mean():.3e}")

        print("\nFirst valid record (φ-only) — quick stats over (nr, nt):")
        stats("Gamma_i_ky", Gamma_i_ky)
        stats("Gamma_e_ky", Gamma_e_ky)
        stats("Qi_ky",      Qi_ky)
        stats("Qe_ky",      Qe_ky)
    else:
        print("No valid (non-empty, correctly-sized) records found to inspect.")

if __name__ == '__main__':
    main()

