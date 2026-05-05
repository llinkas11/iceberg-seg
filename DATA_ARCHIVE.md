# Data archive (GitHub Releases)

Bulk binary data does not fit in the git repo. Tarballs live as assets on GitHub Releases attached to this repo.

> **Status (2026-05-03):** archive build in progress on Bowdoin HPC. Asset URLs and SHA-256 below are placeholders that will be filled in after upload completes. Until then, `download_data.sh` will fail with a clear message.

## Release tag

All bulk-data tarballs are attached to a single release tag: **`archive-v1`**.

Release page: https://github.com/llinkas11/iceberg-seg/releases/tag/archive-v1

## Tarballs

| Tarball | Source on HPC | Approx size | SHA-256 |
|---------|---------------|-------------|---------|
| `chips_v4_clean.tar.zst` | `iceberg-rework/data/v4_clean/` | ~12 GB | `<TBD>` |
| `chips_raw.tar.zst` | `iceberg-rework/data/raw_chips/` (incl. `fisser/`) | ~3 GB | `<TBD>` |
| `chips_smishra.tar.zst` | `smishra/S2-iceberg-areas/chips/` | ~19 GB | `<TBD>` |
| `predictions_area_comparison.tar.zst` | `smishra/S2-iceberg-areas/area_comparison/` | ~2.4 GB | `<TBD>` |
| `predictions_otsu.tar.zst` | `smishra/S2-iceberg-areas/otsu_thresholding/` | ~3.8 GB | `<TBD>` |
| `exp_A0_full.tar.zst` | `iceberg-rework/runs/exp_A0_fisser_lt65_original/` | ~280 MB | `<TBD>` |
| `checkpoints_other.tar.zst` | `.pth` files NOT in `iceberg-rework/canonical_models/` | ~1.5 GB | `<TBD>` |
| `runs_smishra.tar.zst` | `smishra/S2-iceberg-areas/runs/` | ~1.1 GB | `<TBD>` |
| `viz_paper_supporting.tar.zst` | `iceberg-rework/viz/` | ~50 MB | `<TBD>` |
| `viz_annotation_check.tar.zst` | `smishra/{viz_annotation_check, viz_annotation_check2, viz_small_icebergs}/` | ~210 MB | `<TBD>` |
| `unet_implementations.tar.zst` | `smishra/S2UnetPlusPlus/` | ~833 MB | `<TBD>` |
| `train_val_test_v2.tar.zst` | `smishra/train_validate_test_v2/` | ~1.3 GB | `<TBD>` |

Total compressed: ~45 GB.

## Sentinel-2 SAFE downloads NOT included

`smishra/S2-iceberg-areas/sentinel2_downloads/` (49 GB of raw L1C SAFE archives) is intentionally excluded — it can be re-fetched from Copernicus Data Space using the scene IDs recorded in:
- `iceberg-rework/data/v4_clean/manifest.json` (after extracting `chips_v4_clean.tar.zst`)
- `iceberg-rework/reference/v4_test_pools.csv`
- `iceberg-rework/reference/lt65_nulls_selected.csv`

A free Copernicus account is required.

## How to download

The `download_data.sh` script at the repo root pulls all tarballs from the GitHub release and extracts them under `data/`. It uses the `gh` CLI (GitHub's official CLI), which handles auth via your existing `gh auth login` credentials.

```bash
# Install gh once (https://cli.github.com)
brew install gh                              # macOS
gh auth login                                # one-time auth

# Pull all tarballs (~45 GB)
bash download_data.sh

# Or a single tarball
bash download_data.sh chips_v4_clean

# Or a few
bash download_data.sh chips_v4_clean exp_A0_full
```

After download, every tarball's SHA-256 is verified against the table above. Mismatches abort and re-download.

## Repository security note

This is a private repo. The `papers/` subdirectory contains third-party copyrighted PDFs (Fisser et al., CRF reference, etc.) cited in the manuscript. **Do not redistribute or make this repo public** without removing them first.

## Updating the archive

To roll a new tarball or replace an existing one:

```bash
# (on HPC) regenerate the tarball
ssh bowdoin
cd /mnt/research/v.gomezgilyaspik/students/llinkas/bitbucket_archive_staging
tar -I 'zstd -19 -T0' -cf chips_v4_clean.tar.zst -C ../iceberg-rework/data v4_clean
sha256sum chips_v4_clean.tar.zst > chips_v4_clean.sha256

# (locally) upload as a release asset
gh release upload archive-v1 chips_v4_clean.tar.zst --clobber
```

Then update the SHA-256 in this file and `download_data.sh`, commit, push.
