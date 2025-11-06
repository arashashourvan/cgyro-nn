#!/usr/bin/env python3
"""
inspect_cgyro_binary.py
-----------------------
Diagnose CGYRO-style Fortran unformatted binary files (e.g. ky_flux, phi).
Prints record sizes, counts, and guesses likely shape.

Usage:
    python inspect_cgyro_binary.py bin.cgyro.ky_flux
"""

import sys, struct, numpy as np, os

def read_record_headers(fname, max_records=200):
    """Read successive Fortran record headers (both 4B and 8B attempts)."""
    results = []
    with open(fname, "rb") as f:
        offset = 0
        while True:
            header4 = f.read(4)
            if not header4:
                break
            n4 = struct.unpack("i", header4)[0]
            if 0 < n4 < 1e8:
                f.seek(n4 + 4, os.SEEK_CUR)  # skip data + trailer
                results.append((offset, n4, 4))
                offset += 4 + n4 + 4
                continue
            # else try 8-byte header
            f.seek(offset)
            header8 = f.read(8)
            if not header8:
                break
            n8 = struct.unpack("q", header8)[0]
            if 0 < n8 < 1e9:
                f.seek(n8 + 8, os.SEEK_CUR)
                results.append((offset, n8, 8))
                offset += 8 + n8 + 8
                continue
            # if neither worked, bail
            break
            offset += 4
            if len(results) > max_records:
                break
    return results

def inspect(fname):
    sz = os.stat(fname).st_size
    print(f"\nFile: {fname}")
    print(f"Total size: {sz:,} bytes")

    headers = read_record_headers(fname)
    if not headers:
        print("⚠️  No valid Fortran records found.")
        return

    print(f"Detected {len(headers)} records.")
    lens = [n for (_, n, _) in headers]
    types = [t for (_, _, t) in headers]
    unique_types = sorted(set(types))
    print(f"Record marker bytes used: {unique_types}")

    avg_len = np.mean(lens)
    uniq_lens = sorted(set(lens))
    print(f"Record lengths (bytes): min={min(lens)}, max={max(lens)}, "
          f"unique={len(uniq_lens)}; first few={uniq_lens[:5]}")

    # guess float count per record
    rec_floats = uniq_lens[0] // 4
    print(f"→ Likely floats per record: {rec_floats}")
    if rec_floats % 3 == 0:
        print(f"→ That divides by 3 → maybe Nky = {rec_floats // 3}")

    # summarize first few record sizes
    print("\nFirst few record sizes (bytes):")
    for i, (off, n, t) in enumerate(headers[:10]):
        print(f"  Record {i:03d}: offset={off:10d} bytes, length={n:8d}, marker={t}B")

    # detect zero blocks at start
    with open(fname, "rb") as f:
        first = f.read(64)
    if set(first) == {0}:
        print("⚠️  File begins with zeros (possible header/placeholder record).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_cgyro_binary.py <binary_file>")
        sys.exit(1)
    inspect(sys.argv[1])

