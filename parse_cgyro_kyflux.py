#!/usr/bin/env python3
import argparse, numpy as np, struct, os

def read_fortran_record(f, dtype=np.float32):
    """Read one Fortran record (auto-detect 4B or 8B record markers)."""
    header = f.read(4)
    if not header:
        return None
    nbytes = struct.unpack('i', header)[0]

    # sanity check; if this looks wrong, try 8-byte header
    if nbytes <= 0 or nbytes > 10**8:
        # go back and try 8-byte
        f.seek(-4, os.SEEK_CUR)
        header = f.read(8)
        if len(header) < 8:
            return None
        nbytes = struct.unpack('q', header)[0]
        count = nbytes // np.dtype(dtype).itemsize
        data = np.frombuffer(f.read(nbytes), dtype=dtype, count=count)
        f.read(8)  # trailing record marker
        return data
    else:
        count = nbytes // np.dtype(dtype).itemsize
        data = np.frombuffer(f.read(nbytes), dtype=dtype, count=count)
        f.read(4)  # trailing record marker
        return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--phi_npz', required=True)
    ap.add_argument('--merge_into', required=True)
    ap.add_argument('--dky', type=float, required=True)
    ap.add_argument('--k0', type=float, default=0.0)
    ap.add_argument('--force_fortran', type=int, default=1)
    args = ap.parse_args()

    phi_npz = np.load(args.phi_npz)
    T = phi_npz['phi'].shape[0]
    print(f"phi time slices: {T}")

    recs = []
    with open(args.bin, 'rb') as f:
        while True:
            rec = read_fortran_record(f)
            if rec is None:
                break
            recs.append(rec)

    if not recs:
        raise RuntimeError("No Fortran records found — check record marker width or file path.")

    arr = np.vstack(recs)
    print("Loaded ky_flux raw:", arr.shape)

    assert arr.shape[0] == T, f"ky_flux records ({arr.shape[0]}) != phi time slices ({T})"
    nky = arr.shape[1] // 3
    arr = arr.reshape(T, nky, 3)

    ky = args.k0 + np.arange(nky, dtype=np.float32) * args.dky
    flux = (arr * args.dky).sum(axis=1).astype(np.float32)

    out = dict(phi_npz)
    out['ky_flux'] = arr.astype(np.float32)
    out['ky'] = ky
    out['flux'] = flux
    np.savez_compressed(args.merge_into, **out)
    print(f"Saved merged NPZ → {args.merge_into}")
    print(f"Shape ky_flux={arr.shape}, flux={flux.shape}")

if __name__ == "__main__":
    main()

