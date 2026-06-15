# CERT Benchmark Datasets

Downloaded 2026-06-09 for the certflow path-planning project.
Total on-disk: ~228 MB.

---

## 1. METR-LA Traffic Dataset

**Directory:** `metr-la/`

**Source URLs:**
- Time-series + graph (zip): `https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download` (Swiss switch.ch, mirror maintained by the `tsl` torch-spatiotemporal library)
- `adj_mx.pkl`: `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl`
- `distances_la_2012.csv`: `https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv`

**License / Provenance:**
Originally collected from the Los Angeles highway loop-detector network by the California DOT (Caltrans) PeMS system. Released for research by Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018). Public domain traffic sensor data; the DCRNN packaging is MIT-licensed.

**Files:**

| File | Size | Description |
|------|------|-------------|
| `metr_la.h5` | 11 MB | Speed time-series (HDF5/pandas format); 34272 timesteps × 207 sensors, 5-min intervals, float32 |
| `distances_la.csv` | 6.1 MB | Pairwise sensor distances (from switch.ch archive) |
| `distances_la_2012.csv` | 6.1 MB | Same sensor distance matrix from DCRNN repo |
| `sensor_locations_la.csv` | 6.2 KB | Lat/lon for each of the 207 sensors |
| `sensor_ids_la.txt` | 1.5 KB | Sensor ID list |
| `adj_mx.pkl` | 665 KB | Adjacency matrix (list: [sensor_ids, id_map, adj_mx numpy 207×207]) |
| `metr-la.zip` | 13 MB | Original archive (kept for reference) |

**Verified shape:** `block0_values: (34272, 207)` — matches expected ~34272 timesteps × 207 sensors.

**Load snippet:**
```python
import h5py
import numpy as np
import pandas as pd

# Load speed time-series
with h5py.File("data/metr-la/metr_la.h5", "r") as f:
    sensor_ids = [s.decode() for s in f["data/axis0"][:]]   # 207 sensor IDs
    timestamps = f["data/axis1"][:]                           # 34272 int64 timestamps
    speed = f["data/block0_values"][:]                        # (34272, 207) float32
print(f"Speed matrix shape: {speed.shape}")  # (34272, 207)

# Load adjacency matrix
import pickle
with open("data/metr-la/adj_mx.pkl", "rb") as f:
    sensor_ids_list, id_map, adj_mx = pickle.load(f, encoding="latin1")
print(f"Adjacency matrix shape: {adj_mx.shape}")  # (207, 207)

# Load distances
df = pd.read_csv("data/metr-la/distances_la_2012.csv")
print(df.head())
```

---

## 2. PEMS-BAY Traffic Dataset

**Directory:** `pems-bay/`

**Source URL:**
- `https://drive.switch.ch/index.php/s/5NPcgGFAIJ4oFcT/download` (switch.ch mirror, `tsl` library)

**License / Provenance:**
Caltrans PeMS Bay Area sensor data. Released by Li et al. (ICLR 2018), same provenance as METR-LA. Public domain sensor data.

**Files:**

| File | Size | Description |
|------|------|-------------|
| `pems_bay.h5` | 25 MB | Speed time-series; 52128 timesteps × 325 sensors, 5-min intervals, float32 |
| `distances_bay.csv` | 172 KB | Pairwise sensor distances |
| `sensor_locations_bay.csv` | 9.6 KB | Lat/lon for each of the 325 sensors |
| `pems-bay.zip` | 24 MB | Original archive (kept for reference) |

**Verified shape:** `block0_values: (52128, 325)` — matches expected ~52116 timesteps × 325 sensors.

**Load snippet:**
```python
import h5py
import numpy as np

with h5py.File("data/pems-bay/pems_bay.h5", "r") as f:
    sensor_ids = [s.decode() for s in f["data/axis0"][:]]   # 325 sensor IDs
    timestamps = f["data/axis1"][:]                           # 52128 int64 timestamps
    speed = f["data/block0_values"][:]                        # (52128, 325) float32
print(f"Speed matrix shape: {speed.shape}")  # (52128, 325)
```

---

## 3. MovingAI Pathfinding Benchmarks

**Directory:** `movingai/`

**Source URLs:**
All from `https://movingai.com/benchmarks/` (Moving AI Lab, Nathan Sturtevant, University of Denver):

| Archive | URL |
|---------|-----|
| `street-map.zip` | `https://movingai.com/benchmarks/street/street-map.zip` |
| `street-scen.zip` | `https://movingai.com/benchmarks/street/street-scen.zip` |
| `dao-map.zip` | `https://movingai.com/benchmarks/dao/dao-map.zip` |
| `dao-scen.zip` | `https://movingai.com/benchmarks/dao/dao-scen.zip` |
| `maze-map.zip` | `https://movingai.com/benchmarks/maze/maze-map.zip` |
| `maze-scen.zip` | `https://movingai.com/benchmarks/maze/maze-scen.zip` |

**License / Provenance:**
Open Data Commons Attribution License (ODC-By). Original game maps (Dragon Age: Origins, etc.) are included under academic fair-use benchmark licensing. Cite: N. Sturtevant, "Benchmarks for Grid-Based Pathfinding," IEEE TCIAIG 2012.

**Extracted layout:**

| Subdir | Maps | Scen files | Disk |
|--------|------|------------|------|
| `street/` | 90 `.map` | 90 `.scen` | 51 MB |
| `dao/` | 156 `.map` | 156 `.scen` | 29 MB |
| `maze/` | 60 `.map` | 60 `.scen` | 50 MB |

**Map format:** Plain text grid. Header: `type octile`, `height H`, `width W`, `map`, followed by H lines of width W characters (`.` = passable, `@`/`T`/`O` = obstacle).

**Scen format:** Tab-separated, version header `version 1`, then rows: `bucket  map  mapW  mapH  startX  startY  goalX  goalY  optimal_length`.

**Load snippet:**
```python
def load_map(filepath):
    """Load a MovingAI .map file. Returns (height, width, grid_list_of_strings)."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    assert lines[0].strip().startswith("type")
    height = int(lines[1].split()[1])
    width  = int(lines[2].split()[1])
    assert lines[3].strip() == "map"
    grid = [l.rstrip("\n") for l in lines[4:4 + height]]
    return height, width, grid

def load_scen(filepath):
    """Load a MovingAI .scen file. Returns list of dicts."""
    scenarios = []
    with open(filepath, "r") as f:
        next(f)  # skip 'version 1'
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            scenarios.append({
                "bucket": int(parts[0]),
                "map": parts[1],
                "map_w": int(parts[2]),
                "map_h": int(parts[3]),
                "start": (int(parts[4]), int(parts[5])),
                "goal":  (int(parts[6]), int(parts[7])),
                "optimal": float(parts[8]),
            })
    return scenarios

# Example usage
h, w, grid = load_map("data/movingai/street/Berlin_0_256.map")
print(f"Berlin_0_256: {h}x{w}")

scens = load_scen("data/movingai/street/Berlin_0_256.map.scen")
print(f"Loaded {len(scens)} scenarios, e.g. {scens[0]}")

h, w, grid = load_map("data/movingai/dao/arena.map")
print(f"arena: {h}x{w}")

h, w, grid = load_map("data/movingai/maze/maze512-1-0.map")
print(f"maze512-1-0: {h}x{w}")
```

## FoMo (optional — off-road seasonal drift)

Forêt Montmorency dataset (norlab-ulaval), `s3://fomo-dataset` (public, no
credentials). We use the **cost signal only** — GNSS poses (`gt.txt`, TUM),
battery power (`battery_logs.csv`), and weather/snow CSVs — **not** the
9.4 TB of raw lidar/radar/camera. Fetch ~150 MB with:

```bash
cert_env/bin/python scripts/extval/fetch_fomo.py   # -> data/fomo/
```

Source: https://fomo.norlab.ulaval.ca · License: CC BY 4.0 · used in
`docs/results/extended-validation.md` §6.
