"""Load lidar point clouds from common formats into an (N,3) numpy array.

Supported:
    .npy        numpy array, shape (N,>=3)
    .bin        raw float32 stream, KITTI-style (x,y,z,intensity) -> reshaped (N,4)
    .xyz / .txt ASCII rows of "x y z [...]"
    .pcd        ascii PCD (binary PCD not supported here)
    .ply        ascii or binary_little/big_endian; extracts vertex x,y,z
"""

from typing import Optional
import numpy as np


def load_point_cloud(path: str) -> np.ndarray:
    """Load a point cloud, returning float64 array of shape (N, C) with C>=3."""
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    elif lower.endswith(".bin"):
        arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif lower.endswith(".pcd"):
        arr = _load_ascii_pcd(path)
    elif lower.endswith(".ply"):
        arr = _load_ply(path)
    elif lower.endswith((".xyz", ".txt", ".csv")):
        arr = np.loadtxt(path, delimiter=_sniff_delim(path))
    else:
        raise ValueError("unsupported point cloud extension: " + path)

    arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if arr.shape[1] < 3:
        raise ValueError("point cloud needs at least 3 columns (x,y,z)")
    return arr


def _sniff_delim(path: str) -> Optional[str]:
    return "," if path.lower().endswith(".csv") else None


def _load_ascii_pcd(path: str) -> np.ndarray:
    """Minimal ASCII PCD reader (DATA ascii only)."""
    with open(path, "r") as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            if "ascii" not in line:
                raise ValueError("only ascii PCD is supported")
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("malformed PCD: no DATA line")
    rows = [list(map(float, ln.split()[:3])) for ln in lines[data_start:] if ln.strip()]
    return np.asarray(rows, dtype=np.float64)


# PLY scalar type name -> numpy base type. Handles both spellings (float/float32).
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _load_ply(path: str) -> np.ndarray:
    """Read x,y,z from the `vertex` element of a PLY file.

    Supports ascii and binary little/big endian, and arbitrary scalar vertex
    properties (rgb, normals, curvature, ...) -- it builds a structured dtype from
    the header and slices out x,y,z. List properties (e.g. face indices) on the
    vertex element are not supported, but those don't occur for point data.
    """
    with open(path, "rb") as f:
        # Parse the (always-ascii) header line by line.
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError("not a PLY file: " + path)

        fmt = None
        elements = []          # list of (name, count, [(prop_name, type), ...])
        cur = None
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("malformed PLY: no end_header")
            line = raw.decode("ascii", "replace").strip()
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                cur = (tok[1], int(tok[2]), [])
                elements.append(cur)
            elif tok[0] == "property":
                if tok[1] == "list":
                    cur[2].append(("__list__", "list"))
                else:
                    cur[2].append((tok[2], tok[1]))
            elif tok[0] == "end_header":
                break

        vtx = next((e for e in elements if e[0] == "vertex"), None)
        if vtx is None:
            raise ValueError("PLY has no 'vertex' element")
        _, count, props = vtx
        if any(t == "list" for _, t in props):
            raise ValueError("list properties on vertex element are unsupported")

        names = [n for n, _ in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError("PLY vertex element missing '%s'" % axis)

        if fmt == "ascii":
            return _read_ply_ascii(f, count, names)
        endian = "<" if "little" in fmt else ">"
        dtype = np.dtype([(n, endian + _PLY_TYPES[t]) for n, t in props])
        data = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
        return np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)


def _read_ply_ascii(f, count, names) -> np.ndarray:
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    out = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        vals = f.readline().split()
        out[i] = (float(vals[ix]), float(vals[iy]), float(vals[iz]))
    return out
