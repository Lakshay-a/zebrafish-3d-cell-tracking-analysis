#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME="zebrafish-3d-cell-tracking-analysis-source.zip"
ARCHIVE_PATH="$(cd "$REPO_ROOT/.." && pwd)/$ARCHIVE_NAME"
STAGING_DIR="$(mktemp -d)"
PACKAGE_DIR="$STAGING_DIR/zebrafish-3d-cell-tracking-analysis"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

mkdir -p \
  "$PACKAGE_DIR/tracking_utils" \
  "$PACKAGE_DIR/qc_checks" \
  "$PACKAGE_DIR/Feature extraction/plot_scripts"

find "$REPO_ROOT" -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.txt' \
     -o -name '*.yml' -o -name '.env.example' -o -name '.gitignore' \) \
  -exec cp {} "$PACKAGE_DIR/" \;

find "$REPO_ROOT/tracking_utils" -maxdepth 1 -type f -name '*.py' \
  -exec cp {} "$PACKAGE_DIR/tracking_utils/" \;

find "$REPO_ROOT/qc_checks" -maxdepth 1 -type f -name '*.py' \
  -exec cp {} "$PACKAGE_DIR/qc_checks/" \;

find "$REPO_ROOT/Feature extraction" -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.sh' \) \
  -exec cp {} "$PACKAGE_DIR/Feature extraction/" \;

find "$REPO_ROOT/Feature extraction/plot_scripts" -maxdepth 1 -type f -name '*.py' \
  -exec cp {} "$PACKAGE_DIR/Feature extraction/plot_scripts/" \;

for metadata in \
  block_metadata.csv \
  MMP_metadata.csv \
  MMP_analysis_metadata.csv \
  Liraglutide_metadata.csv
do
  cp "$REPO_ROOT/Feature extraction/$metadata" "$PACKAGE_DIR/Feature extraction/"
done

if [ -e "$ARCHIVE_PATH" ]; then
  echo "[ERROR] Archive already exists: $ARCHIVE_PATH"
  echo "Move or rename it before rebuilding."
  exit 1
fi

(
  cd "$STAGING_DIR"
  zip -qr "$ARCHIVE_PATH" "zebrafish-3d-cell-tracking-analysis"
)

echo "[DONE] Created source archive:"
echo "$ARCHIVE_PATH"
