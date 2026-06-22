#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <movie_folder_1> [<movie_folder_2> ...]"
    echo "Example: $0 2026_04_23_M130 2026_04_29_M133"
    exit 1
fi

PY_SCRIPT="make_population_movie.py"
FPS=10
ALPHA=0.4

shopt -s nullglob

for MOVIE in "$@"; do
    ROOT="/Volumes/X10 Pro/Movies/${MOVIE}"
    OUTDIR="${ROOT}/population_movies"
    mkdir -p "$OUTDIR"

    echo "=================================================="
    echo "Making population movies for $MOVIE..."
    echo "=================================================="

    for base_dir in "${ROOT}"/*; do
      [[ -d "$base_dir" ]] || continue

      # Skip if it is the population_movies folder itself
      if [[ "$(basename "$base_dir")" == "population_movies" ]]; then
          continue
      fi

      movie_name="$(basename "$base_dir")"

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

      frames_dir="$(basename "${frames_candidates[0]}")"
      cells_dir="$(basename "${cells_candidates[0]}")"
      out_file="${OUTDIR}/${movie_name}_population.mp4"

      echo "[RUN ] ${movie_name}"
      python "$PY_SCRIPT" \
        --base_dir "$base_dir" \
        --frames_dir "$frames_dir" \
        --cells_dir "$cells_dir" \
        --out "$out_file" \
        --fps "$FPS" \
        --alpha "$ALPHA" || echo "⚠️ Warning: Failed to generate population movie for ${movie_name}!"
    done
done

echo "Done."
