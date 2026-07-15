"""Dependency-free PointCloud2 -> numpy (N,3) xyz parser.

`sensor_msgs_py.point_cloud2` does not exist on ROS2 Foxy (it was added in
Galactic), and Foxy-era helpers return slow per-point tuple generators. This
module parses the PointCloud2 buffer directly with a numpy structured dtype,
so the same code runs on Foxy through Jazzy and is fast (single frombuffer).
"""

import numpy as np

# sensor_msgs/PointField datatype codes -> numpy formats
_DATATYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2",
    5: "i4", 6: "u4", 7: "f4", 8: "f8",
}


def cloud_to_xyz(msg) -> np.ndarray:
    """Extract finite (N,3) float64 x,y,z from a sensor_msgs/PointCloud2."""
    fields = {f.name: f for f in msg.fields}
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise ValueError("PointCloud2 missing field '%s'" % axis)

    endian = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for axis in ("x", "y", "z"):
        f = fields[axis]
        if f.datatype not in _DATATYPES:
            raise ValueError("unsupported datatype %d for field %s" % (f.datatype, axis))
        names.append(axis)
        formats.append(endian + _DATATYPES[f.datatype])
        offsets.append(f.offset)

    dtype = np.dtype({"names": names, "formats": formats,
                      "offsets": offsets, "itemsize": msg.point_step})

    n = msg.width * msg.height
    data = bytes(msg.data)
    if msg.height > 1 and msg.row_step != msg.point_step * msg.width:
        # padded rows: gather row by row
        rows = []
        for r in range(msg.height):
            off = r * msg.row_step
            rows.append(np.frombuffer(data, dtype=dtype, count=msg.width, offset=off))
        pts = np.concatenate(rows)
    else:
        pts = np.frombuffer(data, dtype=dtype, count=n)

    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]
