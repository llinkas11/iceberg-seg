"""One-shot: compare chip-level IC (synth tif vs Fisser source tif) to 10km block IC."""

import os
import numpy as np
import pandas as pd
import rasterio

LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

split = pd.read_csv(f"{LLINKAS}/data/v3_clean/split_log.csv")
lt65_test = split[(split.sza_bin == "sza_lt65") & (split.split == "test")]

prov = pd.read_csv(f"{LLINKAS}/reference/fisser_provenance_audit.csv")
prov["stem"] = prov.global_index.apply(lambda i: f"fisser_{int(i):04d}")
src_map = dict(zip(prov.stem, prov.tif_path))

ic10 = pd.read_csv(f"{LLINKAS}/reference/ic_filter_10km.csv")
print("ic_filter_10km columns:", list(ic10.columns))
print("ic_filter_10km rows:", len(ic10))
print(ic10.head(3).to_string())
print()

rows = []
for _, r in lt65_test.iterrows():
    stem = r.stem
    synth = f"{LLINKAS}/data/raw_chips/fisser/{stem}.tif"
    with rasterio.open(synth) as ds:
        b08_s = ds.read(3).astype(np.float32)
    ic_chip_synth = float((b08_s >= 0.22).mean())
    synth_max = float(b08_s.max())
    src = src_map.get(stem)
    ic_chip_src = None
    src_max = None
    if src and os.path.exists(src):
        with rasterio.open(src) as ds:
            b08_src = ds.read(3).astype(np.float32)
        if b08_src.max() > 100:
            b08_src_ref = b08_src * 1e-4
        else:
            b08_src_ref = b08_src
        ic_chip_src = float((b08_src_ref >= 0.22).mean())
        src_max = float(b08_src_ref.max())
    rows.append({
        "stem": stem, "chip_stem": r.chip_stem,
        "ic_aware_v3": r.ic_aware,
        "ic_chip_synth": ic_chip_synth,
        "ic_chip_src": ic_chip_src,
        "synth_max": synth_max,
        "src_max": src_max,
        "n_icebergs": int(r.n_icebergs),
    })

df = pd.DataFrame(rows)
print("per-chip comparison (first 15 rows):")
print(df.head(15).to_string())
print()
print("summary stats:")
print(df[["ic_aware_v3", "ic_chip_synth", "ic_chip_src"]].describe())
print()
print("agreement: chip-level synth vs chip-level src (diff):")
print((df.ic_chip_synth - df.ic_chip_src).describe())
print()
print("fraction ic_chip_synth >= 0.15:", (df.ic_chip_synth >= 0.15).mean())
print("fraction ic_chip_src   >= 0.15:", (df.ic_chip_src   >= 0.15).mean())
print("fraction ic_aware_v3   >= 0.15:", (df.ic_aware_v3   >= 0.15).mean())
