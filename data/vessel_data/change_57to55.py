
#!/usr/bin/env python3
"""
change_57to55.py  —  v2

• Trim 57‑point hydrodynamic JSON files down to 55 points (drop ω=0 and ω=40).
• Ensure `driftfrc["amp"]` is **6‑DOF** by zero‑padding if the file only
  contains 3 rows (surge, sway, yaw).


Author: Kristian Magnus Røen
Date:   2025-05-05
"""

import sys, json, shutil, datetime as _dt
from pathlib import Path
import numpy as np


def trim57_to_55(arr):
    """Return np.array with every 57‑element axis sliced 1:-1"""
    arr = np.asarray(arr, dtype=np.float32)
    slicer = [slice(None) if dim != 57 else slice(1, -1)
              for dim in arr.shape]
    return arr[tuple(slicer)]


def recurse(obj):
    if isinstance(obj, list):
        try:
            arr = np.asarray(obj, dtype=np.float32)
            if 57 in arr.shape:
                return trim57_to_55(arr).tolist()
        except Exception:
            pass
        return [recurse(x) for x in obj]
    if isinstance(obj, dict):
        return {k: recurse(v) for k, v in obj.items()}
    return obj


def pad_drift_amp(drift_amp):
    """Ensure drift_amp shape is (6, 55, …). Pad zeros on DOF‑axis if needed."""
    arr = np.asarray(drift_amp, dtype=np.float32)
    if arr.shape[0] == 6:
        return arr
    if arr.shape[0] == 3:       # surge, sway, yaw
        zeros = np.zeros((3, *arr.shape[1:]), dtype=arr.dtype)
        arr6 = np.concatenate([arr, zeros], axis=0)
        # copy row 2 (heave) into row 5 (yaw) to match WaveLoad expectations
        arr6[5] = arr6[2]
        arr6[2] = 0.0
        return arr6
    raise RuntimeError(f"Unexpected drift_amp DOF axis length {arr.shape[0]} (wanted 3 or 6)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python change_57to55.py input.json [output.json]")
        sys.exit(1)
    in_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else in_path

    with in_path.open() as f:
        data = json.load(f)

    if "freqs" in data and len(data["freqs"]) == 57:
        data["freqs"] = data["freqs"][1:-1]

    for key in ("A", "B", "Bv"):
        if key in data:
            data[key] = recurse(data[key])

    # driftfrc / forceRAO blocks
    for block in ("forceRAO",):
        if block in data:
            for sub in ("amp", "phase"):
                if sub in data[block]:
                    data[block][sub] = recurse(data[block][sub])

    if "driftfrc" in data and "amp" in data["driftfrc"]:
        trimmed = recurse(data["driftfrc"]["amp"])
        padded = pad_drift_amp(trimmed)
        data["driftfrc"]["amp"] = padded.tolist()
        if "phase" in data["driftfrc"]:
            data["driftfrc"]["phase"] = recurse(data["driftfrc"]["phase"])

    # backup if overwriting
    if out_path == in_path:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = in_path.with_suffix(in_path.suffix + f".bak_{ts}")
        shutil.copy2(in_path, backup)
        print("Backup saved to", backup)

    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    print("Trimmed & padded JSON written to", out_path)


if __name__ == "__main__":
    main()
