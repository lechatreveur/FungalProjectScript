#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:29:05 2025

@author: user
"""

# plotting_cells.py
import math
import os
from typing import Iterable, List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt


def plot_cells_grid(
    data: Union[str, pd.DataFrame],
    *,
    time_col: str = "time_point",
    value_col: str = "pattern_score_norm",
    id_col: str = "cell_id",
    channel_col: Optional[str] = "channel",
    channel: Optional[str] = None,
    nrows: int = 10,
    ncols: int = 10,
    sharey: bool = True,
    figsize: tuple = (20, 20),
    cell_ids: Optional[Iterable[Union[int, str]]] = None,
    sort_time: bool = True,
    paginate: bool = True,
    save_dir: Optional[str] = None,
    filename_prefix: str = "cells_grid",
    dpi: int = 150,
) -> List[plt.Figure]:
    """
    Plot value vs time for each cell in 10x10 grids (100 cells per figure).

    Parameters
    ----------
    data : str | pd.DataFrame
        Path to CSV or a DataFrame with at least [time_col, value_col, id_col].
    time_col : str
        Column representing time (e.g., "time_point").
    value_col : str
        Column to plot on y-axis (e.g., "pattern_score_norm").
    id_col : str
        Identifier column for cells (e.g., "cell_id").
    channel_col : str | None
        Column indicating channel. If None, no channel filtering.
    channel : str | None
        If provided, filter to rows where channel_col == channel.
    nrows, ncols : int
        Grid size per figure. Default 10x10.
    sharey : bool
        Share y-axis across subplots per page.
    figsize : tuple
        Figure size per page (in inches).
    cell_ids : iterable | None
        If provided, only plot these cell_ids (order respected).
    sort_time : bool
        Sort points by time within each cell.
    paginate : bool
        If True, create multiple figures when > nrows*ncols cells.
    save_dir : str | None
        If provided, save each page as PNG into this directory.
    filename_prefix : str
        Prefix for saved filenames.
    dpi : int
        DPI for saved figures.

    Returns
    -------
    List[matplotlib.figure.Figure]
        The list of figures created (one per page).
    """
    # --- load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    required = {time_col, value_col, id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- optional channel filter
    if channel is not None:
        if channel_col is None:
            raise ValueError("channel_col must be set when filtering by channel.")
        if channel_col not in df.columns:
            raise ValueError(f"channel_col '{channel_col}' not in DataFrame.")
        df = df[df[channel_col] == channel]

    # --- clean types
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col, value_col, id_col])

    # --- determine which cells to plot
    all_cells = pd.unique(df[id_col])
    if cell_ids is not None:
        # keep order user provided; ensure existent
        cell_set = set(all_cells)
        selected = [cid for cid in cell_ids if cid in cell_set]
    else:
        selected = sorted(all_cells)

    if len(selected) == 0:
        raise ValueError("No cells to plot after filtering/selection.")

    per_page = nrows * ncols
    num_pages = 1 if not paginate else math.ceil(len(selected) / per_page)

    figs: List[plt.Figure] = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # --- paging
    for page in range(num_pages):
        start = page * per_page
        end = start + per_page
        batch = selected[start:end]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey)
        # axes is 2D array when nrows,ncols > 1
        axes_flat = axes.ravel()

        # plot each cell
        for ax, cid in zip(axes_flat, batch):
            sub = df[df[id_col] == cid]
            if sort_time:
                sub = sub.sort_values(time_col)

            # Plot; default Matplotlib style/no explicit colors
            ax.plot(sub[time_col].values, sub[value_col].values, marker=".", linewidth=1)
            ax.set_title(f"{id_col}={cid}", fontsize=8)
            ax.set_xlabel(time_col, fontsize=7)
            ax.set_ylabel(value_col, fontsize=7)
            ax.tick_params(axis="both", which="both", labelsize=7)

        # hide unused axes on the last page
        for ax in axes_flat[len(batch):]:
            ax.axis("off")

        fig.tight_layout()
        figs.append(fig)

        # save if requested
        if save_dir is not None:
            fname = f"{filename_prefix}_page{page+1:03d}.png"
            fig.savefig(os.path.join(save_dir, fname), dpi=dpi, bbox_inches="tight")

    return figs
