#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Movies/2025_12_31_M92"

targets=(
  "$ROOT/A14-YES-1t-FBFBF-6_F0"
  "$ROOT/A14-YES-1t-FBFBF-6_F1"
  "$ROOT/A14-YES-1t-FBFBF-6_F2"
)

rename_one() {
  local p="$1"
  local dir base new new_path

  dir="$(dirname "$p")"
  base="$(basename "$p")"

  new="$base"
  new="${new//(20min)/}"
  new="${new//()/}"

  [[ "$new" == "$base" ]] && return 0

  new_path="$dir/$new"
  if [[ -e "$new_path" ]]; then
    echo "[SKIP exists] $p -> $new_path"
    return 0
  fi

  echo "[RENAME] $p -> $new_path"
  mv -n "$p" "$new_path"
}

for d in "${targets[@]}"; do
  [[ -d "$d" ]] || { echo "[SKIP] not a dir: $d"; continue; }
  echo "[SCAN] $d"

  # Pass 1: items containing "(20min)"
  {
    find "$d" -depth -print0 \
      | grep -z -F '(20min)' \
      | while IFS= read -r -d '' p; do
          rename_one "$p"
        done
  } || true

  # Pass 2: items containing "()"
  {
    find "$d" -depth -print0 \
      | grep -z -F '()' \
      | while IFS= read -r -d '' p; do
          rename_one "$p"
        done
  } || true
done

echo "Done."
