import numpy as np
import pathlib
import csv

rows = []
for npz_path in sorted(pathlib.Path("logs").glob("*/latency_results.npz")):
    run_name = npz_path.parent.name
    parts = run_name.rsplit("_", 1)
    level, hz = parts[0], int(parts[1].replace("hz", ""))
    data = np.load(npz_path)
    row = {"control_level": level, "control_hz": hz}
    for key in data.files:
        arr = data[key]  # already in milliseconds
        row[f"{key}_mean_ms"] = float(np.mean(arr))
        row[f"{key}_p99_ms"] = float(np.percentile(arr, 99))
    rows.append(row)

rows.sort(key=lambda r: (r["control_level"], r["control_hz"]))

out_path = pathlib.Path("logs/sweep_summary.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {out_path}")
for r in rows:
    print(f"{r['control_level']:15s} {r['control_hz']:4d}Hz  total_mean={r['total_ms_mean_ms']:.3f}ms")
