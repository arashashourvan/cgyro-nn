#!/usr/bin/env python3
"""
check_kyflux_dimensions.py
--------------------------
Check if a CGYRO ky_flux Fortran binary matches expected dimensions:
  (n_species=2, m=3, n_field=2, n_t, n_time)

Usage:
  python check_kyflux_dimensions.py --bin ./bin.cgyro.ky_flux --nt 16
"""

import argparse, os, struct
import numpy as np

def read_fortran_records(path):
    """Return list of raw float arrays from a Fortran-unformatted file."""
    recs = []
    with open(path, "rb") as f:
        while True:
            # try 4-byte header
            pos = f.tell()
            h4 = f.read(4)
            if not h4:
                break
            n4 = struct.unpack("i", h4)[0]
            if 0 < n4 < 1e9:
                buf = f.read(n4)
                f.read(4)
                recs.append(np.frombuffer(buf, dtype=np.float32))
                continue
            # try 8-byte header
            f.seek(pos)
            h8 = f.read(8)
            if not h8:
                break
            n8 = struct.unpack("q", h8)[0]
            if 0 < n8 < 1e12:
                buf = f.read(n8)
                f.read(8)
                recs.append(np.frombuffer(buf, dtype=np.float32))
                continue
            # invalid header → stop
            break
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--nt", type=int, default=16, help="Number of ky points (try 16 or 15)")
    args = ap.parse_args()

    ns, nm, nf = 2, 3, 2
    base = ns * nm * nf  # 12
    path = args.bin
    size_bytes = os.stat(path).st_size

    print(f"\nFile: {path}")
    print(f"Total file size: {size_bytes:,} bytes")

    recs = read_fortran_records(path)
    if not recs:
        print("❌ No valid Fortran records found.")
        return

    print(f"Detected {len(recs)} record(s).")
    floats_per_rec = [r.size for r in recs]
    print(f"Floats per record: {floats_per_rec[:5]}{'...' if len(floats_per_rec)>5 else ''}")

    # try multi-record scheme (one record per time)
    if len(set(floats_per_rec)) == 1:
        per_rec = floats_per_rec[0]
        if per_rec % (base*args.nt) == 0:
            ntime = per_rec // (base*args.nt)
            print(f"✅ Multi-record layout fits: (2,3,2,{args.nt},{ntime}) per record")
        else:
            print(f"❌ Record size {per_rec} not divisible by 12*nt={base*args.nt}")

    # try single-record scheme
    total_floats = sum(floats_per_rec)
    if total_floats % (base*args.nt) == 0:
        ntime = total_floats // (base*args.nt)
        print(f"✅ Single-record layout fits: (2,3,2,{args.nt},{ntime}) total")
    else:
        print(f"❌ Total float count {total_floats} not divisible by 12*nt={base*args.nt}")

    # show any zero/empty records
    empties = sum(r.size == 0 for r in recs)
    if empties:
        print(f"⚠️  {empties} empty records skipped.")

if __name__ == "__main__":
    main()

