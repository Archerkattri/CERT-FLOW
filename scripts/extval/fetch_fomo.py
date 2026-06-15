"""Download ONLY the FoMo cost signal (poses + power + weather), not the 9.4 TB
of raw lidar/radar/camera. Public bucket, no credentials. ~120 MB total."""
import json, urllib.request, pathlib, sys
BASE = "https://fomo-dataset.s3.amazonaws.com"
OUT = pathlib.Path("data/fomo"); OUT.mkdir(parents=True, exist_ok=True)
COST_CSVS = ["current_left.csv","current_right.csv","voltage_left.csv","voltage_right.csv",
             "velocity_left.csv","velocity_right.csv","battery_logs.csv",
             "meteo_data.csv","snow_data.csv","emlid_metadata.csv"]
sizes = json.load(open("/tmp/fomo_sizes.json"))
got = miss = 0; total = 0
for date, trajs in sizes.items():
    for traj in trajs:
        sess = f"data/{date}/{traj}/"
        dst = OUT/date/traj; dst.mkdir(parents=True, exist_ok=True)
        for rel in ["gt.txt"] + [f"metadata/{c}" for c in COST_CSVS]:
            (dst/rel).parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(f"{BASE}/{sess}{rel}", dst/rel)
                total += (dst/rel).stat().st_size; got += 1
            except Exception as e:
                miss += 1
        print(f"  {date}/{traj}: done", flush=True)
print(f"FoMo cost-signal download: {got} files OK, {miss} missing, {total/1e6:.0f} MB")
