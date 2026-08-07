import os
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, base_movie_root: Path, max_bytes: int = 0, max_files: int = 0):
        """
        base_movie_root: root under which <exp>/<film>/{CellCrops_,PopulationFrames_}<film>
            directories live. As of the cache_root config change, this is a
            consolidated cache folder (e.g. X10 Pro's drive root, sibling to
            Movies/), not scattered through the movie data tree.
        max_bytes: soft cap for the combined LOGICAL size of every render cache
            under base_movie_root. 0 disables enforcement.
        max_files: soft cap for the total NUMBER of cached files. This is the
            cap that matters on filesystems with large block/cluster sizes
            (X10 Pro measures 512KB-1MB per file): a byte cap alone doesn't
            protect against overhead from many small files. 0 disables.
        Both enforced on-demand via enforce_cache_limit(), not continuously.
        """
        self.base_root = base_movie_root
        self.max_bytes = max_bytes
        self.max_files = max_files

    def clear_population_cache(self, exp: str, film: str) -> int:
        cache_dir = self.base_root / exp / film / f"PopulationFrames_{film}"
        return self._clear_dir(cache_dir)

    def clear_cell_crops_cache(self, exp: str, film: str) -> int:
        cache_dir = self.base_root / exp / film / f"CellCrops_{film}"
        return self._clear_dir(cache_dir)

    def clear_all_caches_for_film(self, exp: str, film: str) -> int:
        return self.clear_population_cache(exp, film) + self.clear_cell_crops_cache(exp, film)

    def _clear_dir(self, cache_dir: Path) -> int:
        cleared_count = 0
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if f.is_file() and (f.suffix.lower() in ['.jpg', '.jpeg', '.png'] or f.name.startswith("._")):
                    try:
                        f.unlink()
                        cleared_count += 1
                    except Exception as e:
                        logger.warning("Failed to unlink cache file %s: %s", f, e)
        return cleared_count

    def _iter_cache_files(self):
        if not self.base_root.exists():
            return
        for exp_dir in self.base_root.iterdir():
            if not exp_dir.is_dir():
                continue
            for film_dir in exp_dir.iterdir():
                if not film_dir.is_dir():
                    continue
                for cache_dir in film_dir.glob("*"):
                    if not cache_dir.is_dir():
                        continue
                    if not (cache_dir.name.startswith("CellCrops_") or cache_dir.name.startswith("PopulationFrames_")):
                        continue
                    for f in cache_dir.rglob("*"):
                        if f.is_file():
                            yield f

    def enforce_cache_limit(self) -> dict:
        """
        Defensive cap so the render cache can't grow unbounded: if the total
        FILE COUNT or combined logical byte size across every CellCrops_*/
        PopulationFrames_* directory under base_root exceeds either cap,
        delete the least-recently-modified cache files first until back under
        BOTH caps. File count is checked first since it's the more meaningful
        limit on a large-block-size filesystem (see cache_max_files).

        Cheap to call (e.g. once per app startup); not a continuous watcher.
        Returns a small report dict for logging.
        """
        report = {
            "scanned_files": 0, "scanned_bytes": 0,
            "removed_files": 0, "removed_bytes": 0, "enforced": False,
        }
        if (not self.max_bytes and not self.max_files) or not self.base_root.exists():
            return report

        entries = []  # (mtime, size, path)
        total_bytes = 0
        for f in self._iter_cache_files():
            try:
                st = f.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, f))
            total_bytes += st.st_size

        total_files = len(entries)
        report["scanned_files"] = total_files
        report["scanned_bytes"] = total_bytes

        over_files = bool(self.max_files) and total_files > self.max_files
        over_bytes = bool(self.max_bytes) and total_bytes > self.max_bytes
        if not (over_files or over_bytes):
            return report

        report["enforced"] = True
        entries.sort(key=lambda e: e[0])  # oldest first

        target_files = self.max_files if self.max_files else total_files
        target_bytes = self.max_bytes if self.max_bytes else total_bytes

        removed_files = 0
        removed_bytes = 0
        idx = 0
        while idx < len(entries) and (
            (total_files - removed_files) > target_files
            or (total_bytes - removed_bytes) > target_bytes
        ):
            _mtime, size, path = entries[idx]
            idx += 1
            try:
                path.unlink()
                removed_files += 1
                removed_bytes += size
            except OSError as e:
                logger.warning("Failed to unlink %s during cache eviction: %s", path, e)

        report["removed_files"] = removed_files
        report["removed_bytes"] = removed_bytes
        logger.info(
            "Render cache eviction: was %d files / %.1f MB (caps: %s files, %s MB) -- removed %d files / %.1f MB",
            total_files, total_bytes / 1e6,
            self.max_files or "none", (self.max_bytes / 1e6) if self.max_bytes else "none",
            removed_files, removed_bytes / 1e6,
        )
        return report
