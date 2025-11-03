#!/usr/bin/env python3
"""
parse_cgyro_phi.py
------------------
Parse CGYRO complex potential binaries into a compact NPZ suitable for NN training.

Supports two layouts per time slice (choose via auto-detect or flags):
1) RAW contiguous (no record markers): a simple concatenation of T slices,
   each of shape [Nr, Ntheta, Ntor, 2] with float32 values for [Re, Im].
2) Fortran unformatted records: each time slice is one record framed by
   4- or 8-byte record markers (leading/trailing). Endianness auto-detected.

Features
- Streaming: does not load the whole file into RAM
- Temporal stride (downsample frames): --stride_t
- Spatial downsampling: --ds_r, --ds_theta, --ds_tor
- Optional FFT over theta and toroidal dims to move to spectral indices
- Optional parsing of a transport text file to attach [Qi, Qe, Gamma]

Usage examples
--------------
# Parse a known RAW file with Nr=324, Ntheta=1, Ntor=16 (like your sample):
python parse_cgyro_phi.py --bin bin.cgyro.kxky_phi --Nr 324 --Ntheta 1 --Ntor 16 \
    --force_raw 1 --out phi_only.npz

# Try auto-detect (attempt RAW first, then Fortran records):
python parse_cgyro_phi.py --bin out.cgyro.phi --Nr 324 --Ntheta 24 --Ntor 16 --out phi_only.npz

# Include transport (Qi,Qe,Gamma), temporal stride, spatial ds, and FFT to ky,n:
python parse_cgyro_phi.py --bin out.cgyro.phi --Nr 324 --Ntheta 24 --Ntor 16 \
    --stride_t 2 --ds_r 2 --fft_theta 1 --fft_tor 1 \
    --transport out.cgyro.transport --out my_run_phi_flux.npz

Output
------
NPZ with:
  - phi:  float32 [T', Nr', Ntheta'(or ky), Ntor'(or n), 2]
  - flux: float32 [T', 3] if --transport provided (Qi, Qe, Gamma)
"""

import argparse, os, sys, struct
import numpy as np

def sizeof(path):
    return os.stat(path).st_size

def try_raw_mode(path, Nr, Nt, Nn):
    """Check if file size is divisible by payload size (RAW contiguous test)."""
    payload_bytes = Nr*Nt*Nn*2*4  # float32, [Re,Im]
    sz = sizeof(path)
    if sz % payload_bytes == 0:
        T = sz // payload_bytes
        return True, int(T), payload_bytes
    return False, 0, payload_bytes

def autodetect_rec_bytes_endian(fh, payload_bytes):
    """Try to infer 4/8-byte record markers and endianness from first record."""
    pos0 = fh.tell()
    head = fh.read(8)
    fh.seek(pos0)
    if len(head) < 4:
        return None, None
    for rec_bytes in (4,8):
        if len(head) < rec_bytes:
            continue
        for endian in ('<','>'):
            fmt = ('<I' if endian=='<' else '>I') if rec_bytes==4 else ('<Q' if endian=='<' else '>Q')
            try:
                (n,) = struct.unpack(fmt, head[:rec_bytes])
            except struct.error:
                continue
            if n == payload_bytes:
                return rec_bytes, endian
    return None, None

def read_fortran_record(fh, payload_bytes, rec_bytes, endian):
    """Read a single Fortran unformatted record of a known payload size."""
    marker_fmt = ('<I' if endian=='<' else '>I') if rec_bytes==4 else ('<Q' if endian=='<' else '>Q')
    head = fh.read(rec_bytes)
    if not head or len(head) < rec_bytes:
        return None  # EOF
    (n_pre,) = struct.unpack(marker_fmt, head)
    data = fh.read(payload_bytes)
    if len(data) < payload_bytes:
        return None
    tail = fh.read(rec_bytes)
    if not tail or len(tail) < rec_bytes:
        return None
    # Could validate n_post here if desired:
    # (n_post,) = struct.unpack(marker_fmt, tail)
    return data

def parse_transport_txt(path):
    """
    Minimal transport parser: tries to find Qi, Qe, Gamma columns by header tokens.
    If no headers, falls back to first three numeric columns.
    Returns [T,3] float32 array in order [Qi, Qe, Gamma].
    """
    lines = []
    with open(path, 'r') as f:
        for ln in f:
            ls = ln.strip()
            if not ls or ls.startswith('#'):
                continue
            lines.append(ls)
    if not lines:
        raise RuntimeError("Empty transport file")
    # Detect header
    header_idx = 0
    hdr_tokens = lines[header_idx].lower().split()
    # Heuristic: if all tokens are numeric, there's probably no header
    try:
        [float(tok) for tok in hdr_tokens]
        # numeric -> treat as data, synthesize headers
        headers = ['c%d'%i for i in range(len(hdr_tokens))]
        data = np.array([list(map(float, ln.split())) for ln in lines], dtype=float)
    except ValueError:
        # we have a header
        headers = hdr_tokens
        data = np.array([list(map(float, ln.split())) for ln in lines[1:]], dtype=float)

    def find_col(keys):
        for k in keys:
            if k in headers: return headers.index(k)
        return None

    ci = find_col(['qi','q_i','qi(ion)','qi(gyro-bohm)','qi(gyrob)'])
    ce = find_col(['qe','q_e','qe(elec)','qe(gyro-bohm)','qe(gyrob)'])
    cg = find_col(['gamma','gam','gamm','Γ'])
    if any(c is None for c in [ci,ce,cg]):
        # fallback to first three columns
        ci, ce, cg = 0, 1, 2

    flux = np.stack([data[:,ci], data[:,ce], data[:,cg]], axis=1).astype(np.float32)
    return flux

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True, help='Path to CGYRO φ binary')
    ap.add_argument('--Nr', type=int, required=True)
    ap.add_argument('--Ntheta', type=int, required=True)
    ap.add_argument('--Ntor', type=int, required=True)
    ap.add_argument('--force_raw', type=int, default=0, help='1 to force RAW contiguous mode')
    ap.add_argument('--force_fortran', type=int, default=0, help='1 to force Fortran-record mode')
    ap.add_argument('--rec_bytes', type=int, default=0, help='Record-marker bytes (4 or 8); 0=auto')
    ap.add_argument('--endian', type=str, default='auto', help="'<' little, '>' big, or 'auto'")
    ap.add_argument('--stride_t', type=int, default=1, help='Temporal stride (keep every Nth slice)')
    ap.add_argument('--max_slices', type=int, default=0, help='Maximum slices to read (0 = all)')
    ap.add_argument('--ds_r', type=int, default=1, help='Downsample factor in radial dimension')
    ap.add_argument('--ds_theta', type=int, default=1, help='Downsample factor in theta dimension')
    ap.add_argument('--ds_tor', type=int, default=1, help='Downsample factor in toroidal dimension')
    ap.add_argument('--fft_theta', type=int, default=0, help='Apply FFT over theta dim (1=yes)')
    ap.add_argument('--fft_tor', type=int, default=0, help='Apply FFT over toroidal dim (1=yes)')
    ap.add_argument('--transport', type=str, default='', help='Optional path to out.cgyro.transport')
    ap.add_argument('--out', required=True, help='Output NPZ path')
    args = ap.parse_args()

    Nr, Nt, Nn = args.Nr, args.Ntheta, args.Ntor
    payload_elems = Nr*Nt*Nn*2
    payload_bytes = payload_elems * 4

    # Decide mode
    mode = None
    T = 0

    if args.force_raw and args.force_fortran:
        print("Choose only one of --force_raw or --force_fortran.", file=sys.stderr)
        sys.exit(2)

    if args.force_raw:
        ok, T, _ = try_raw_mode(args.bin, Nr, Nt, Nn)
        if not ok:
            print("RAW mode forced but file size not divisible by slice size.", file=sys.stderr)
            sys.exit(2)
        mode = 'raw'
    elif args.force_fortran:
        mode = 'fortran'
    else:
        ok, T, _ = try_raw_mode(args.bin, Nr, Nt, Nn)
        mode = 'raw' if ok else 'fortran'

    phi_slices = []
    keep = 0

    if mode == 'raw':
        sz = os.stat(args.bin).st_size
        T = sz // payload_bytes
        mm = np.memmap(args.bin, dtype=np.float32, mode='r', shape=(T*payload_elems,))
        for t in range(T):
            if (t % args.stride_t) != 0:
                continue
            start = t*payload_elems
            arr = np.array(mm[start:start+payload_elems], dtype=np.float32)
            if arr.size != payload_elems:
                break
            arr = arr.reshape(Nr, Nt, Nn, 2)
            # Spatial DS
            if args.ds_r > 1:     arr = arr[::args.ds_r, :, :, :]
            if args.ds_theta > 1: arr = arr[:, ::args.ds_theta, :, :]
            if args.ds_tor > 1:   arr = arr[:, :, ::args.ds_tor, :]
            # FFTs
            if args.fft_theta:
                c = arr[...,0] + 1j*arr[...,1]
                c = np.fft.fftshift(np.fft.fft(c, axis=1), axes=1)
                arr = np.stack([c.real, c.imag], axis=-1).astype(np.float32)
            if args.fft_tor:
                c = arr[...,0] + 1j*arr[...,1]
                c = np.fft.fftshift(np.fft.fft(c, axis=2), axes=2)
                arr = np.stack([c.real, c.imag], axis=-1).astype(np.float32)
            phi_slices.append(arr)
            keep += 1
            if args.max_slices and keep >= args.max_slices:
                break
    else:
        with open(args.bin, 'rb') as fh:
            rec_bytes = args.rec_bytes if args.rec_bytes in (4,8) else None
            endian = None if args.endian == 'auto' else args.endian
            if rec_bytes is None or endian is None:
                rb, ed = autodetect_rec_bytes_endian(fh, payload_bytes)
                rec_bytes = rec_bytes or (rb if rb else 4)
                endian = endian or (ed if ed else '<')
            t = 0
            while True:
                raw = read_fortran_record(fh, payload_bytes, rec_bytes, endian)
                if raw is None:
                    break
                if (t % args.stride_t) != 0:
                    t += 1; continue
                arr = np.frombuffer(raw, dtype=np.float32, count=payload_elems)
                if arr.size != payload_elems:
                    break
                arr = arr.reshape(Nr, Nt, Nn, 2)
                # Spatial DS
                if args.ds_r > 1:     arr = arr[::args.ds_r, :, :, :]
                if args.ds_theta > 1: arr = arr[:, ::args.ds_theta, :, :]
                if args.ds_tor > 1:   arr = arr[:, :, ::args.ds_tor, :]
                # FFTs
                if args.fft_theta:
                    c = arr[...,0] + 1j*arr[...,1]
                    c = np.fft.fftshift(np.fft.fft(c, axis=1), axes=1)
                    arr = np.stack([c.real, c.imag], axis=-1).astype(np.float32)
                if args.fft_tor:
                    c = arr[...,0] + 1j*arr[...,1]
                    c = np.fft.fftshift(np.fft.fft(c, axis=2), axes=2)
                    arr = np.stack([c.real, c.imag], axis=-1).astype(np.float32)
                phi_slices.append(arr)
                keep += 1
                t += 1
                if args.max_slices and keep >= args.max_slices:
                    break

    if not phi_slices:
        print("No slices parsed. Check dimensions or flags.", file=sys.stderr)
        sys.exit(2)

    phi = np.stack(phi_slices, axis=0).astype(np.float32)

    # Optional transport
    flux = None
    if args.transport:
        try:
            flux = parse_transport_txt(args.transport)
        except Exception as e:
            print(f"Warning: transport parse failed: {e}", file=sys.stderr)
            flux = None

    if flux is not None:
        Tm = min(phi.shape[0], flux.shape[0])
        if Tm < phi.shape[0]:
            phi = phi[:Tm]
        if Tm < flux.shape[0]:
            flux = flux[:Tm]

    if flux is None:
        np.savez_compressed(args.out, phi=phi)
        print(f"Wrote {args.out} with phi shape {phi.shape}")
    else:
        np.savez_compressed(args.out, phi=phi, flux=flux.astype(np.float32))
        print(f"Wrote {args.out} with phi shape {phi.shape} and flux shape {flux.shape}")

if __name__ == "__main__":
    main()
