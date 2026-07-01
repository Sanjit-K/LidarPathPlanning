#!/usr/bin/env python3
"""Downsample the large Open3D apartment PLY (~30M double-precision points) to a
manageable .npy for the discretization pipeline.

Reads the binary PLY body in sequential chunks (memory-capped) and keeps every
Nth point, so it works on a memory-constrained machine. Saves an (M,3) float32
array.
"""
import sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "test-data/aligned_low_apartment/apt.ply"
DST = sys.argv[2] if len(sys.argv) > 2 else "test-data/aligned_low_apartment/apt_sub.npy"
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 8

dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")])

with open(SRC, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += f.read(64)
    count = int([l.split()[-1] for l in hdr.decode("ascii", "replace").splitlines()
                 if l.startswith("element vertex")][0])
    data_start = hdr.find(b"end_header\n") + len(b"end_header\n")
    f.seek(data_start)
    print("vertices: %d  stride: %d  -> ~%d points" % (count, STRIDE, count // STRIDE),
          flush=True)

    CHUNK = 2_000_000
    parts = []
    read = 0
    phase = 0
    while read < count:
        n = min(CHUNK, count - read)
        block = np.fromfile(f, dtype=dt, count=n)
        if block.shape[0] == 0:
            break
        # uniform stride across the whole file, tracking phase between chunks
        idx = np.arange(phase, block.shape[0], STRIDE)
        phase = (phase + STRIDE * len(idx) - block.shape[0]) % STRIDE
        sel = block[idx]
        parts.append(np.stack([sel["x"], sel["y"], sel["z"]], axis=-1).astype(np.float32))
        read += block.shape[0]
        print("  read %d / %d (%.0f%%)" % (read, count, 100 * read / count), flush=True)

xyz = np.concatenate(parts, axis=0)
xyz = xyz[np.isfinite(xyz).all(axis=1)]
np.save(DST, xyz)
print("saved %s : %d points" % (DST, xyz.shape[0]), flush=True)
print("extent x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]" % (
    xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max(),
    xyz[:, 2].min(), xyz[:, 2].max()), flush=True)
