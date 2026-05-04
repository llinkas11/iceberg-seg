#!/usr/bin/env bash
# Download tier-3 tarballs from GitHub Releases and extract into data/.
# Usage:
#   bash download_data.sh                          # all tarballs
#   bash download_data.sh chips_v4_clean           # one tarball
#   bash download_data.sh chips_v4_clean exp_A0_full

set -euo pipefail

# 1. Resolve repo root and create data/ at the same level as the script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 2. Release config + tarball SHA-256 registry.
#    SHAs are filled in after the tarballs are uploaded as release assets.
GH_REPO="llinkas11/iceberg-seg"
RELEASE_TAG="archive-v1"

declare -A TARBALLS_SHA
TARBALLS_SHA[chips_v4_clean]="<TBD>"
TARBALLS_SHA[chips_raw]="<TBD>"
TARBALLS_SHA[chips_smishra]="<TBD>"
TARBALLS_SHA[predictions_area_comparison]="<TBD>"
TARBALLS_SHA[predictions_otsu]="<TBD>"
TARBALLS_SHA[exp_A0_full]="<TBD>"
TARBALLS_SHA[checkpoints_other]="<TBD>"
TARBALLS_SHA[runs_smishra]="<TBD>"
TARBALLS_SHA[viz_paper_supporting]="<TBD>"
TARBALLS_SHA[viz_annotation_check]="<TBD>"
TARBALLS_SHA[unet_implementations]="<TBD>"
TARBALLS_SHA[train_val_test_v2]="<TBD>"

# 3. Verify gh CLI is installed and authenticated.
if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: 'gh' (GitHub CLI) is required. Install via:"
    echo "  brew install gh   (macOS)"
    echo "  https://cli.github.com  (other platforms)"
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: 'gh' is not authenticated. Run: gh auth login"
    exit 1
fi

# 4. Pick which tarballs to fetch.
if [ "$#" -eq 0 ]; then
    NAMES=("${!TARBALLS_SHA[@]}")
else
    NAMES=("$@")
fi

# 5. Download + verify + extract each tarball.
for name in "${NAMES[@]}"; do
    expected_sha="${TARBALLS_SHA[$name]:-}"
    if [ -z "$expected_sha" ] || [ "$expected_sha" = "<TBD>" ]; then
        echo "ERROR: SHA-256 for $name not yet set in DATA_ARCHIVE.md / download_data.sh"
        echo "       Archive build is in progress; rerun once SHAs land."
        exit 2
    fi
    echo "=== $name ==="
    archive="$name.tar.zst"
    if [ -f "$archive" ]; then
        echo "  already downloaded: $archive"
    else
        echo "  fetching from gh release $RELEASE_TAG"
        gh release download "$RELEASE_TAG" -R "$GH_REPO" -p "$archive"
    fi
    actual_sha=$(shasum -a 256 "$archive" | awk '{print $1}')
    if [ "$actual_sha" != "$expected_sha" ]; then
        echo "ERROR: SHA-256 mismatch for $archive"
        echo "  expected: $expected_sha"
        echo "  actual:   $actual_sha"
        exit 3
    fi
    echo "  sha256 OK"
    echo "  extracting"
    tar -I zstd -xf "$archive"
    echo "  done"
done

echo ""
echo "All requested tarballs downloaded and extracted into: $DATA_DIR"
