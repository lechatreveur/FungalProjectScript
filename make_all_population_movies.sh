#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/X10 Pro/Movies/2026_01_16_M96"
PY_SCRIPT="make_population_movie.py"

FPS=10
ALPHA=0.4

# Optional: where to write outputs (default: inside each movie folder)
OUTDIR="${ROOT}/population_movies"
mkdir -p "$OUTDIR"

shopt -s nullglob

for base_dir in "${ROOT}"/*; do
  [[ -d "$base_dir" ]] || continue

  movie_name="$(basename "$base_dir")"

  # Find Frames_* and TrackedCells_* inside this movie folder
  frames_candidates=("${base_dir}"/Frames_*)
  cells_candidates=("${base_dir}"/TrackedCells_*)

  if (( ${#frames_candidates[@]} == 0 )); then
    echo "[SKIP] ${movie_name}: no Frames_* directory found"
    continue
  fi
  if (( ${#cells_candidates[@]} == 0 )); then
    echo "[SKIP] ${movie_name}: no TrackedCells_* directory found"
    continue
  fi

  # If multiple, pick the first (common case is exactly one)
  frames_dir="$(basename "${frames_candidates[0]}")"
  cells_dir="$(basename "${cells_candidates[0]}")"

  out_file="${base_dir}/${movie_name}_population.mp4"

  # Or if using a global output folder:
  out_file="${OUTDIR}/${movie_name}_population.mp4"

  echo "[RUN ] ${movie_name}"
  echo "      base_dir=${base_dir}"
  echo "      frames_dir=${frames_dir}"
  echo "      cells_dir=${cells_dir}"
  echo "      out=${out_file}"

  python "$PY_SCRIPT" \
    --base_dir "$base_dir" \
    --frames_dir "$frames_dir" \
    --cells_dir "$cells_dir" \
    --out "$out_file" \
    --fps "$FPS" \
    --alpha "$ALPHA"
done

echo "Done."
