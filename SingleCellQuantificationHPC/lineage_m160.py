#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve M160 linkage tracks into a mother-forks-into-daughters lineage.

Why this exists
---------------
`sequence_linkage.json` carries a `lineage` key, but it is **empty**. Division
is encoded implicitly instead: a mother and her daughters share the same local
cell id in every film *before* the division, then diverge. So the same
`(film, local_cell_id)` is legitimately claimed by two or more `global_cell_id`s.

Treating that as duplication and keeping only the first claimant collapses the
lineage: 18 of 950 selected cells ended up with no films at all and only 860
kept all seven, though the linkage gives every one of them seven.

The model
---------
Build a trie over each cell's sequence of local ids across the fluorescence
films. A node where the trie branches is a division. The path from the root to
the first branch is the **mother**; each branch below it is a **daughter**; a
daughter that branches again is itself a mother further down.

Every `(film, local_cell_id)` therefore belongs to exactly ONE segment, so no
measurement is counted twice, and the segments form a tree that the explorer can
draw as a path that forks.

Identity
--------
A segment is named for the film and local id it starts at:

    <sequence>__<film>_<local_cell_id>        e.g. 5_1_N1_F0__5_1_N1_FL1_F0_22

Deterministic, readable, and it says where the segment begins. Every segment
also carries `member_gids`, the original `global_cell_id`s that run through it,
so nothing is lost for traceability (P12).

Not every contested pair is a division. Pairs whose tracks are identical
throughout are true duplicates and collapse to one segment; pairs that diverge
and then re-converge are a linkage fault and are reported, not silently merged.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]


def fl_tracks(exp_dir, seq, keep_gids=None):
    """gid -> [(film, local_cell_id), ...] over the FL films, in film order."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))[seq]
    films = linkage["films"]
    fl = [(i, f) for i, f in enumerate(films) if "FL" in f]
    out = {}
    for gid, track in linkage["global_cells"].items():
        if keep_gids is not None and gid not in keep_gids:
            continue
        path = []
        for i, f in fl:
            lc = track[i] if i < len(track) else -1
            if lc and lc > 0:
                path.append((f, int(lc)))
        if path:
            out[gid] = path
    return out


def resolve(tracks):
    """Split tracks into segments at their branch points.

    Returns (segments, edges, notes) where
      segments: seg_id -> dict(points=[(film, lc)...], member_gids=[...],
                               parent=seg_id|None, depth=int)
      edges:    [(parent_seg_id, child_seg_id), ...]
      notes:    diagnostics that need a human (re-converging tracks, duplicates)
    """
    notes = {"identical_gids": [], "reconverging": [], "excluded_gids": []}

    # true duplicates: identical whole track -> one representative
    by_path = defaultdict(list)
    for gid, path in tracks.items():
        by_path[tuple(path)].append(gid)
    for path, gids in by_path.items():
        if len(gids) > 1:
            notes["identical_gids"].append(sorted(gids))

    # Re-converging tracks are a linkage fault: two paths diverge and then land
    # back on the same local cell. A cell does not un-divide, so this is either
    # a mis-link or a tracking error. Excluded from the lineage and reported —
    # left in, they put the shared tail in two segments at once, which breaks
    # the one-point-one-segment invariant everything downstream relies on.
    items = sorted(by_path.items())
    bad_paths = set()
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            p1, p2 = items[a][0], items[b][0]
            n = min(len(p1), len(p2))
            same = [i for i in range(n) if p1[i] == p2[i]]
            diff = [i for i in range(n) if p1[i] != p2[i]]
            if same and diff and max(same) > min(diff):
                notes["reconverging"].append(
                    (sorted(items[a][1])[0], sorted(items[b][1])[0]))
                bad_paths.add(p1)
                bad_paths.add(p2)
    if bad_paths:
        notes["excluded_gids"] = sorted(g for p in bad_paths for g in by_path[p])
        by_path = {p: g for p, g in by_path.items() if p not in bad_paths}

    # trie over the distinct paths
    segments, edges = {}, []

    def seg_id(point):
        film, lc = point
        return f"{film}_{lc}"

    def build(paths, start, parent, depth):
        """paths: list of (path, gids). All share path[:start]."""
        if not paths:
            return
        # walk forward while every path agrees
        i = start
        while True:
            vals = {p[i] for p, _ in paths if i < len(p)}
            ended = [1 for p, _ in paths if i >= len(p)]
            if len(vals) != 1 or ended:
                break
            i += 1
        points = [paths[0][0][k] for k in range(start, i)]
        if points:
            sid = seg_id(points[0])
            members = sorted({g for _, gs in paths for g in gs})
            segments[sid] = dict(points=points, member_gids=members,
                                 parent=parent, depth=depth)
            if parent is not None:
                edges.append((parent, sid))
            parent, depth = sid, depth + 1
        # branch
        groups = defaultdict(list)
        for p, gs in paths:
            if i < len(p):
                groups[p[i]].append((p, gs))
        if len(groups) == 1 and points == []:
            return   # nothing to do, avoid a loop
        for _, grp in sorted(groups.items()):
            build(grp, i, parent, depth)

    build([(list(p), gs) for p, gs in sorted(by_path.items())], 0, None, 0)
    return segments, edges, notes


def resolve_experiment(exp_dir, keep_gids=None, seqs=SEQS):
    """-> (point_to_segment, segments, edges, notes) for the whole experiment."""
    all_seg, all_edges = {}, []
    all_notes = {"identical_gids": [], "reconverging": [], "excluded_gids": []}
    point_map = {}
    for seq in seqs:
        tracks = fl_tracks(exp_dir, seq, keep_gids)
        if not tracks:
            continue
        segs, edges, notes = resolve(tracks)
        for sid, s in segs.items():
            s["sequence"] = seq
            all_seg[sid] = s
            for pt in s["points"]:
                point_map[pt] = sid
        all_edges += edges
        for k in all_notes:
            all_notes[k] += notes[k]
    return point_map, all_seg, all_edges, all_notes
