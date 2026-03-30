
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class LinearWindowFit:
    start: int
    end: int  # inclusive
    length: int
    slope: float
    intercept: float
    ssr: float  # sum of squared residuals

def _best_increasing_linear_window_1d(
    y: np.ndarray,
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
) -> Optional[LinearWindowFit]:
    y = np.asarray(y).astype(float)
    T = y.shape[0]
    if T < min_len:
        return None

    t = np.arange(T, dtype=float)
    pref_y = np.concatenate([[0.0], np.cumsum(y)])
    pref_y2 = np.concatenate([[0.0], np.cumsum(y * y)])
    pref_ty = np.concatenate([[0.0], np.cumsum(t * y)])

    best: Optional[LinearWindowFit] = None

    for i in range(0, T - min_len + 1):
        for j in range(i + min_len - 1, T):
            n = j - i + 1
            Sx = n * (n - 1) / 2.0
            Sxx = (n - 1) * n * (2 * n - 1) / 6.0

            Sy = pref_y[j + 1] - pref_y[i]
            Syy = pref_y2[j + 1] - pref_y2[i]
            Sty = pref_ty[j + 1] - pref_ty[i]
            Sxy = Sty - i * Sy

            denom = n * Sxx - Sx * Sx
            if denom <= 0:
                continue

            a = (n * Sxy - Sx * Sy) / denom
            if a <= 0:
                continue

            b = (Sy - a * Sx) / n

            if enforce_targets:
                y_start = b
                y_end   = a * (n - 1) + b
                if not (abs(y_start - start_target) <= start_tol and
                        abs(y_end   - end_target)   <= end_tol):
                    continue

            ssr = Syy + (a * a) * Sxx + n * (b * b) - 2 * a * Sxy - 2 * b * Sy + 2 * a * b * Sx

            if (best is None) or (ssr < best.ssr):
                best = LinearWindowFit(start=i, end=j, length=n, slope=a, intercept=b, ssr=ssr)

    return best

def find_best_increasing_linear_window(
    trajectory: Union[np.ndarray, List[float]],
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
) -> Optional[LinearWindowFit]:
    y = np.asarray(trajectory).astype(float).ravel()
    return _best_increasing_linear_window_1d(
        y, min_len=min_len,
        enforce_targets=enforce_targets,
        start_target=start_target, end_target=end_target,
        start_tol=start_tol, end_tol=end_tol
    )

def scan_many_trajectories(
    data: Union[np.ndarray, List[List[float]]],
    min_len: int = 21,
    axis: int = 0,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
) -> List[Optional[LinearWindowFit]]:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return [find_best_increasing_linear_window(
            arr, min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol
        )]

    if axis == 1:
        arr = arr.T

    T, N = arr.shape
    results: List[Optional[LinearWindowFit]] = []
    for k in range(N):
        results.append(_best_increasing_linear_window_1d(
            arr[:, k], min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol
        ))
    return results

def plot_trajectory_with_fit(
    y: Union[np.ndarray, List[float]],
    fit: LinearWindowFit,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    y = np.asarray(y).astype(float).ravel()
    t = np.arange(len(y), dtype=float)

    n = fit.length
    xw = np.arange(n, dtype=float)
    yw = fit.slope * xw + fit.intercept

    plt.figure()
    plt.plot(t, y, label="raw")
    tw = np.arange(fit.start, fit.end + 1, dtype=float)
    plt.plot(tw, yw, label=f"fit (slope={fit.slope:.3g})")
    plt.axvspan(fit.start, fit.end, alpha=0.1)
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel("time (index)")
    plt.ylabel("value")
    if show:
        plt.show()

def analyze_pattern_score_norm(
    pattern_score_norm: Any,
    min_len: int = 21,
    axis: int = 0,
    which: Optional[int] = None,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
) -> Dict[str, Any]:
    arr = np.asarray(pattern_score_norm, dtype=float)
    result: Dict[str, Any] = {"fits": [], "plotted_index": None, "fit": None}

    if arr.ndim == 1:
        fit = find_best_increasing_linear_window(
            arr, min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol,
        )
        result["fits"] = [fit]
        result["fit"] = fit
        if fit is not None:
            plot_trajectory_with_fit(arr, fit, title="pattern_score_norm (1D)")
        return result

    fits = scan_many_trajectories(
        arr, min_len=min_len, axis=axis,
        enforce_targets=enforce_targets,
        start_target=start_target, end_target=end_target,
        start_tol=start_tol, end_tol=end_tol,
    )
    result["fits"] = fits

    idx_to_plot = None
    if which is not None:
        if 0 <= which < len(fits) and fits[which] is not None:
            idx_to_plot = which
    else:
        valid = [(i, f.ssr) for i, f in enumerate(fits) if f is not None]
        if valid:
            idx_to_plot = min(valid, key=lambda z: z[1])[0]

    if idx_to_plot is not None:
        result["plotted_index"] = idx_to_plot
        fit = fits[idx_to_plot]
        result["fit"] = fit
        plot_arr = arr if axis == 0 else arr.T
        plot_trajectory_with_fit(plot_arr[:, idx_to_plot], fit, title=f"pattern_score_norm[{idx_to_plot}]")

    return result

def _trajectory_from_df(
    df_all: pd.DataFrame,
    cell_id: int,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
) -> np.ndarray:
    sub = (
        df_all.loc[df_all["cell_id"] == cell_id, [time_col, value_col]]
        .dropna()
        .sort_values(time_col)
    )
    return sub[value_col].to_numpy()

def fit_cell_increasing_window(
    df_all: pd.DataFrame,
    cell_id: int,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
):
    y = _trajectory_from_df(df_all, cell_id, value_col=value_col, time_col=time_col)
    if y.size == 0:
        return None, y
    fit = find_best_increasing_linear_window(
        y, min_len=min_len,
        enforce_targets=enforce_targets,
        start_target=start_target, end_target=end_target,
        start_tol=start_tol, end_tol=end_tol,
    )
    return fit, y

def plot_cell_with_increasing_fit(
    df_all: pd.DataFrame,
    cell_id: int,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
    min_len: int = 21,
    title_prefix: str = "Cell",
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
):
    fit, y = fit_cell_increasing_window(
        df_all, cell_id, value_col=value_col, time_col=time_col, min_len=min_len,
        enforce_targets=enforce_targets,
        start_target=start_target, end_target=end_target,
        start_tol=start_tol, end_tol=end_tol,
    )
    if fit is None:
        print(f"[{cell_id}] no valid >= {min_len}-point increasing window found (given target constraints).")
        return
    plot_trajectory_with_fit(y, fit, title=f"{title_prefix} {cell_id}")

def scan_cells_summary(
    df_all: pd.DataFrame,
    cell_ids,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for cid in cell_ids:
        fit, y = fit_cell_increasing_window(
            df_all, cid, value_col=value_col, time_col=time_col, min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol,
        )
        if fit is None:
            rows.append(dict(
                cell_id=cid, valid=False, length=np.nan, slope=np.nan, intercept=np.nan,
                ssr=np.nan, start=np.nan, end=np.nan
            ))
        else:
            rows.append(dict(
                cell_id=cid, valid=True, length=fit.length, slope=fit.slope,
                intercept=fit.intercept, ssr=fit.ssr, start=fit.start, end=fit.end
            ))
    return pd.DataFrame(rows)

def plot_top_k_by_ssr(
    df_all: pd.DataFrame,
    summary_df: pd.DataFrame,
    k: int = 12,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
):
    top = summary_df[summary_df["valid"]].sort_values("ssr", ascending=True).head(k)
    for cid in top["cell_id"]:
        plot_cell_with_increasing_fit(
            df_all, cid, value_col=value_col, time_col=time_col, min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol,
        )


from matplotlib.widgets import SpanSelector, Button
import matplotlib.pyplot as plt

def _ols_fit_with_ssr(yw: np.ndarray):
    n = yw.size
    if n < 2:
        return None, None, None
    Sx  = n * (n - 1) / 2.0
    Sxx = (n - 1) * n * (2 * n - 1) / 6.0
    Sy  = float(np.sum(yw))
    Syy = float(np.sum(yw * yw))
    Sxy = float(np.sum(np.arange(n, dtype=float) * yw))

    denom = n * Sxx - Sx * Sx
    if denom <= 0:
        return None, None, None

    a = (n * Sxy - Sx * Sy) / denom
    b = (Sy - a * Sx) / n
    ssr = Syy + (a * a) * Sxx + n * (b * b) - 2 * a * Sxy - 2 * b * Sy + 2 * a * b * Sx
    return a, b, ssr
from matplotlib.widgets import SpanSelector, Button
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Module-level registry to prevent GC of widgets created inside helpers
try:
    _LIVE_WIDGETS
except NameError:
    _LIVE_WIDGETS = []


def plot_cell_with_increasing_fit_gui(
    df_all: pd.DataFrame,
    cell_id: int,
    summary_df: pd.DataFrame = None,
    value_col: str = "pattern_score_norm",
    time_col: str = "time_point",
    min_len: int = 21,
    enforce_targets: bool = False,
    start_target: float = 0.1,
    end_target: float = 0.4,
    start_tol: float = 0.05,
    end_tol: float = 0.05,
    title_prefix: str = "Cell",
    block: bool = False,   # NEW: block until window closes if True
):
    """
    Interactive GUI to drag-select a window, fit y = a*x + b (x=0..n-1) with a>0,
    optionally enforce start/end targets, and (optionally) update summary_df.

    Returns
    -------
    dict with keys:
      - 'fit'  : LinearWindowFit or None  (populated after window closes)
      - 'start': int or None
      - 'end'  : int or None
      - 'updated_summary': summary_df (possibly modified)
      - 'fig'  : the matplotlib Figure (always returned; keep a reference!)
    """
    # ---- data ----
    y = _trajectory_from_df(df_all, cell_id, value_col=value_col, time_col=time_col)
    if y.size == 0:
        print(f"[{cell_id}] empty trajectory.")
        return {"fit": None, "start": None, "end": None, "updated_summary": summary_df, "fig": None}

    t = np.arange(len(y), dtype=float)

    # ---- figure ----
    fig, ax = plt.subplots()
    plt.ion()  # ensure interactive
    ax.plot(t, y, label="raw")
    ax.set_title(f"{title_prefix} {cell_id} — drag to select window")
    ax.set_xlabel("time (index)")
    ax.set_ylabel(value_col)
    ax.legend()

    # Ensure toolbar doesn’t steal events
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            if hasattr(tb, "mode"): tb.mode = ""
            if hasattr(tb, "pan"):  tb.pan();  tb.pan()
            if hasattr(tb, "zoom"): tb.zoom(); tb.zoom()
    except Exception:
        pass

    # Bring to front (Qt-friendly)
    try:
        w = fig.canvas.manager.window
        w.raise_(); w.activateWindow()
    except Exception:
        pass

    # Layout for Save button
    plt.subplots_adjust(bottom=0.28)
    save_ax = plt.axes([0.72, 0.06, 0.22, 0.12])
    btn_save = Button(save_ax, "Save")

    # ---- state ----
    state = {"i": None, "j": None, "fit": None, "line": None, "span": None}

    def _ols_fit_with_ssr(yw: np.ndarray):
        n = yw.size
        if n < 2:
            return None, None, None
        Sx  = n * (n - 1) / 2.0
        Sxx = (n - 1) * n * (2 * n - 1) / 6.0
        Sy  = float(np.sum(yw))
        Syy = float(np.sum(yw * yw))
        Sxy = float(np.sum(np.arange(n, dtype=float) * yw))
        denom = n * Sxx - Sx * Sx
        if denom <= 0:
            return None, None, None
        a = (n * Sxy - Sx * Sy) / denom
        b = (Sy - a * Sx) / n
        ssr = Syy + (a * a) * Sxx + n * (b * b) - 2 * a * Sxy - 2 * b * Sy + 2 * a * b * Sx
        return a, b, ssr

    def draw_fit(i: int, j: int):
        # Clear previous overlays
        if state["line"] is not None:
            try: state["line"].remove()
            except Exception: pass
            state["line"] = None
        if state["span"] is not None:
            try: state["span"].remove()
            except Exception: pass
            state["span"] = None

        n = j - i + 1
        yw = y[i:j+1]
        a, b, ssr = _ols_fit_with_ssr(yw)
        if a is None:
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            return None
        if a <= 0:
            ax.set_title(f"{title_prefix} {cell_id} — slope <= 0, reselect")
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            return None

        # Optional start/end constraints on fitted line
        if enforce_targets:
            y_start = b
            y_end   = a * (n - 1) + b
            ok = (abs(y_start - start_target) <= start_tol) and (abs(y_end - end_target) <= end_tol)
            if not ok:
                ax.set_title(f"{title_prefix} {cell_id} — outside target bounds, reselect")
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                return None

        # Draw fitted line over absolute indices
        xw = np.arange(n, dtype=float)
        yw_fit = a * xw + b
        tw = np.arange(i, j+1, dtype=float)
        line, = ax.plot(tw, yw_fit, label="fit")
        span = ax.axvspan(i, j, alpha=0.1)
        ax.legend()

        fit_obj = LinearWindowFit(start=i, end=j, length=n, slope=a, intercept=b, ssr=float(ssr))
        state["i"], state["j"], state["fit"] = i, j, fit_obj
        state["line"], state["span"] = line, span
        ax.set_title(f"{title_prefix} {cell_id} — selected [{i}:{j}] slope={a:.3g}, SSR={ssr:.3g}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        return fit_obj

    def onselect(xmin, xmax):
        # convert float x to int indices [i, j]
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        i = int(np.floor(lo))
        j = int(np.ceil(hi))
        i = max(0, min(i, len(y)-1))
        j = max(0, min(j, len(y)-1))
        if j - i + 1 < min_len:
            ax.set_title(f"{title_prefix} {cell_id} — select >= {min_len} points")
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            return
        draw_fit(i, j)

    # Robust selector (no blitting)
    span_selector = SpanSelector(
        ax, onselect, direction="horizontal",
        useblit=False, interactive=True, grab_range=8
    )
    try:
        span_selector.set_active(True)
    except Exception:
        pass

    def on_save(event):
        fit = state["fit"]
        if fit is None:
            print("No valid selection to save.")
            return
        if summary_df is None:
            print("Saved (no summary_df provided to update).")
            return
        row = dict(
            cell_id=cell_id,
            valid=True,
            length=fit.length,
            slope=fit.slope,
            intercept=fit.intercept,
            ssr=fit.ssr,
            start=fit.start,
            end=fit.end,
        )
        if "cell_id" in summary_df.columns and (summary_df["cell_id"] == cell_id).any():
            summary_df.loc[summary_df["cell_id"] == cell_id, row.keys()] = list(row.values())
        else:
            summary_df.loc[len(summary_df)] = row
        print(f"Saved selection for cell {cell_id}: [{fit.start}, {fit.end}] slope={fit.slope:.4g} SSR={fit.ssr:.4g}")

    btn_save.on_clicked(on_save)

    # ---------- PERSISTENT REFERENCES (prevents GC) ----------
    fig._span_selector = span_selector
    fig._save_button = btn_save
    _LIVE_WIDGETS.append((fig, span_selector, btn_save))
    # Clean up registry on close
    def _on_close(evt):
        # fill result on close
        result["fit"] = state["fit"]
        result["start"] = state["i"]
        result["end"] = state["j"]
        # remove from registry
        try:
            for k, tup in enumerate(list(_LIVE_WIDGETS)):
                if tup[0] is fig:
                    _LIVE_WIDGETS.pop(k)
                    break
        except Exception:
            pass
    fig.canvas.mpl_connect("close_event", _on_close)
    # --------------------------------------------------------

    # return object (filled on close if block=True, or later when closed)
    result = {"fit": None, "start": None, "end": None, "updated_summary": summary_df, "fig": fig}

    # Show window; block if requested
    plt.show(block=block)

    # If blocking, we can return the filled result; if non-blocking, the
    # result['fit'] etc. will be populated when the window closes.
    return result

# helpers_gui.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button

# module-level registry to hold live widgets
_LIVE_WIDGETS = []

def span_select_gui(
    y, x=None, title="Drag to select; click Save", min_len=1, on_save=None,
    block=False,   # if True, show() will block
):
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x).ravel()

    fig, ax = plt.subplots()
    ax.plot(x, y, label="raw")
    ax.set_title(title)
    ax.legend(loc="best")

    plt.subplots_adjust(bottom=0.28)
    save_ax = plt.axes([0.72, 0.06, 0.22, 0.12])
    btn = Button(save_ax, "Save")

    # ensure toolbar not intercepting
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None and hasattr(tb, "mode"):
            tb.mode = ""
    except Exception:
        pass

    state = {"i": None, "j": None}

    def _onselect(xmin, xmax):
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        i = int(np.searchsorted(x, lo, side="left"))
        j = int(np.searchsorted(x, hi, side="right")) - 1
        i = max(0, min(i, len(x)-1))
        j = max(0, min(j, len(x)-1))
        if j - i + 1 < min_len:
            ax.set_title(f"Select at least {min_len} points (got {j - i + 1})")
            fig.canvas.draw_idle()
            return
        state["i"], state["j"] = i, j
        ax.set_title(f"Selected [{i}:{j}] (N={j-i+1})")
        fig.canvas.draw_idle()

    ss = SpanSelector(
        ax, _onselect, direction="horizontal",
        useblit=False, interactive=True, grab_range=8
    )

    def _on_save(event):
        i, j = state["i"], state["j"]
        if i is None or j is None:
            print("No valid selection to save.")
            return
        if on_save is not None:
            try:
                on_save(i, j, x[i:j+1], y[i:j+1])
            except Exception as e:
                print("on_save error:", e)
        print(f"Saved selection: i={i}, j={j}, N={j - i + 1}")

    btn.on_clicked(_on_save)

    # ---------- keep references alive ----------
    fig._span_selector = ss
    fig._save_button = btn
    _LIVE_WIDGETS.append((fig, ss, btn))
    def _on_close(evt):
        # remove from registry on close
        for k, tup in enumerate(list(_LIVE_WIDGETS)):
            if tup[0] is fig:
                _LIVE_WIDGETS.pop(k)
                break
    fig.canvas.mpl_connect("close_event", _on_close)
    # -------------------------------------------

    # Block or not depending on your preference
    plt.show(block=block)

    # return fig so caller can also keep a handle if desired
    return fig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button

# Keep widgets alive across function scope (prevents GC)
try:
    _LIVE_WIDGETS
except NameError:
    _LIVE_WIDGETS = []

def review_cells_gui(
    df_all,
    cell_ids,
    summary_df=None,
    value_col="pattern_score_norm",
    time_col="time_point",
    min_len=21,
    enforce_targets=True,
    start_target=0.0,
    end_target=0.9,
    start_tol=0.1,
    end_tol=0.5,
    title_prefix="Cell",
    block=False,   # True => modal; False => non-blocking (keep coding)
):
    """
    Walk through `cell_ids`. For each cell:
      1) show raw data and the (auto) fitted line (if available),
      2) allow manual drag selection to refit,
      3) Save => write/overwrite row in summary_df and move to next cell.

    Returns
    -------
    dict with keys:
      - 'fig': Figure (keep a handle alive in caller)
      - 'summary': summary_df (same object, possibly updated)
      - 'last_fit': the last LinearWindowFit (or None)
      - 'last_cell': last cell_id viewed
    """
    # --- internal helpers we rely on from your module ---
    # _trajectory_from_df(df_all, cell_id, value_col, time_col) -> np.ndarray
    # find_best_increasing_linear_window(y, min_len, enforce_targets, start_target, end_target, start_tol, end_tol)
    # LinearWindowFit dataclass

    # Shared state
    state = {
        "k": 0,               # index into cell_ids
        "cell_ids": list(cell_ids),
        "y": None,            # current trajectory
        "t": None,            # x indices
        "fit": None,          # current LinearWindowFit
        "line": None,         # fitted line artist
        "span": None,         # highlight artist
        "dirty": False,       # whether a manual selection has updated the fit
    }

    if summary_df is None:
        import pandas as pd
        summary_df = pd.DataFrame(columns=["cell_id","valid","length","slope","intercept","ssr","start","end"])

    # ---- figure & axes ----
    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlabel("time (index)")
    ax.set_ylabel(value_col)

    # Give space for two buttons (Save / Next)
    plt.subplots_adjust(bottom=0.30)
    save_ax = plt.axes([0.56, 0.06, 0.18, 0.14])
    next_ax = plt.axes([0.78, 0.06, 0.18, 0.14])
    btn_save = Button(save_ax, "Save")
    btn_next = Button(next_ax, "Skip/Next")

    # toolbar sanity
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            if hasattr(tb, "mode"): tb.mode = ""
            if hasattr(tb, "pan"):  tb.pan();  tb.pan()
            if hasattr(tb, "zoom"): tb.zoom(); tb.zoom()
    except Exception:
        pass

    # Bring window to front (Qt)
    try:
        w = fig.canvas.manager.window
        w.raise_(); w.activateWindow()
    except Exception:
        pass

    # ---- fitting utils ----
    def _ols_fit_with_ssr(yw: np.ndarray):
        n = yw.size
        if n < 2:
            return None, None, None
        Sx  = n * (n - 1) / 2.0
        Sxx = (n - 1) * n * (2 * n - 1) / 6.0
        Sy  = float(np.sum(yw))
        Syy = float(np.sum(yw * yw))
        Sxy = float(np.sum(np.arange(n, dtype=float) * yw))
        denom = n * Sxx - Sx * Sx
        if denom <= 0:
            return None, None, None
        a = (n * Sxy - Sx * Sy) / denom
        b = (Sy - a * Sx) / n
        ssr = Syy + (a * a) * Sxx + n * (b * b) - 2 * a * Sxy - 2 * b * Sy + 2 * a * b * Sx
        return a, b, ssr

    def _apply_fit(i: int, j: int):
        """Compute fit on y[i:j] with constraints; draw overlays."""
        # clear old overlays
        if state["line"] is not None:
            try: state["line"].remove()
            except Exception: pass
            state["line"] = None
        if state["span"] is not None:
            try: state["span"].remove()
            except Exception: pass
            state["span"] = None

        y = state["y"]
        n = j - i + 1
        yw = y[i:j+1]
        a, b, ssr = _ols_fit_with_ssr(yw)
        if a is None or a <= 0:
            ax.set_title(f"{title_prefix} {state['cell_ids'][state['k']]} — invalid slope; reselect")
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            state["fit"] = None
            return

        if enforce_targets:
            y_start = b
            y_end   = a * (n - 1) + b
            ok = (abs(y_start - start_target) <= start_tol) and (abs(y_end - end_target) <= end_tol)
            if not ok:
                ax.set_title(f"{title_prefix} {state['cell_ids'][state['k']]} — outside targets; reselect")
                fig.canvas.draw_idle(); fig.canvas.flush_events()
                state["fit"] = None
                return

        # draw
        xw = np.arange(n, dtype=float)
        yw_fit = a * xw + b
        tw = np.arange(i, j+1, dtype=float)
        line, = ax.plot(tw, yw_fit, label="fit")
        span = ax.axvspan(i, j, alpha=0.1)
        ax.legend()

        from dataclasses import dataclass
        # use your LinearWindowFit
        fit_obj = LinearWindowFit(start=i, end=j, length=n, slope=a, intercept=b, ssr=float(ssr))
        state["fit"] = fit_obj
        state["line"], state["span"] = line, span
        state["dirty"] = True
        ax.set_title(f"{title_prefix} {state['cell_ids'][state['k']]} — selected [{i}:{j}] slope={a:.3g}, SSR={ssr:.3g}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()

    # ---- cell navigation ----
    def _load_cell(k: int):
        """Load cell k: draw raw and auto-fit if available."""
        ax.cla()
        cell_id = state["cell_ids"][k]
        y = _trajectory_from_df(df_all, cell_id, value_col=value_col, time_col=time_col)
        t = np.arange(len(y), dtype=float)
        state["y"], state["t"], state["fit"], state["line"], state["span"], state["dirty"] = y, t, None, None, None, False

        ax.plot(t, y, label="raw")
        ax.set_xlabel("time (index)")
        ax.set_ylabel(value_col)
        ax.legend()

        # try auto-fit
        auto = find_best_increasing_linear_window(
            y, min_len=min_len,
            enforce_targets=enforce_targets,
            start_target=start_target, end_target=end_target,
            start_tol=start_tol, end_tol=end_tol
        )
        if auto is not None:
            # paint auto fit
            i, j = auto.start, auto.end
            xw = np.arange(auto.length, dtype=float)
            yw_fit = auto.slope * xw + auto.intercept
            tw = np.arange(i, j+1, dtype=float)
            line, = ax.plot(tw, yw_fit, label="fit (auto)")
            span = ax.axvspan(i, j, alpha=0.1)
            state["fit"] = auto
            state["line"], state["span"] = line, span
            ax.set_title(f"{title_prefix} {cell_id} — auto-fit [{i}:{j}] slope={auto.slope:.3g}, SSR={auto.ssr:.3g}")
        else:
            ax.set_title(f"{title_prefix} {cell_id} — no valid auto-fit; drag to select (≥{min_len})")

        fig.canvas.draw_idle(); fig.canvas.flush_events()

    # initial load
    if len(state["cell_ids"]) == 0:
        print("No cell IDs provided.")
        return {"fig": fig, "summary": summary_df, "last_fit": None, "last_cell": None}
    _load_cell(state["k"])

    # ---- selector ----
    def _onselect(xmin, xmax):
        y = state["y"]
        if y is None or len(y) == 0:
            return
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        i = int(np.floor(lo))
        j = int(np.ceil(hi))
        i = max(0, min(i, len(y)-1))
        j = max(0, min(j, len(y)-1))
        if j - i + 1 < min_len:
            ax.set_title(f"{title_prefix} {state['cell_ids'][state['k']]} — select ≥ {min_len} points")
            fig.canvas.draw_idle(); fig.canvas.flush_events()
            return
        _apply_fit(i, j)

    span_selector = SpanSelector(
        ax, _onselect, direction="horizontal",
        useblit=False, interactive=True, grab_range=8
    )
    try: span_selector.set_active(True)
    except Exception: pass

    # ---- actions ----
    def _write_summary_current():
        cell_id = state["cell_ids"][state["k"]]
        fit = state["fit"]
        if fit is None:
            print("No valid fit to save; please drag-select a window.")
            return False
        row = dict(
            cell_id=cell_id,
            valid=True,
            length=fit.length,
            slope=fit.slope,
            intercept=fit.intercept,
            ssr=fit.ssr,
            start=fit.start,
            end=fit.end,
        )
        import pandas as pd
        if "cell_id" in summary_df.columns and (summary_df["cell_id"] == cell_id).any():
            summary_df.loc[summary_df["cell_id"] == cell_id, row.keys()] = list(row.values())
        else:
            summary_df.loc[len(summary_df)] = row
        print(f"Saved: cell {cell_id} [{fit.start}, {fit.end}] slope={fit.slope:.4g} SSR={fit.ssr:.4g}")
        return True

    def _advance(delta=1):
        # move to next/prev cell; close if past the end
        state["k"] += delta
        if state["k"] >= len(state["cell_ids"]):
            print("All cells reviewed. Closing.")
            plt.close(fig)
            return
        if state["k"] < 0:
            state["k"] = 0
        _load_cell(state["k"])

    def _on_save(event):
        if _write_summary_current():
            _advance(+1)

    def _on_next(event):
        # skip without saving (or if there is no fit yet)
        _advance(+1)

    btn_save.on_clicked(_on_save)
    btn_next.on_clicked(_on_next)

    # ---- persistent refs to avoid GC ----
    fig._span_selector = span_selector
    fig._btn_save = btn_save
    fig._btn_next = btn_next
    _LIVE_WIDGETS.append((fig, span_selector, btn_save, btn_next))

    result = {"fig": fig, "summary": summary_df, "last_fit": None, "last_cell": None}
    def _on_close(evt):
        # record last viewed
        result["last_cell"] = state["cell_ids"][state["k"]] if 0 <= state["k"] < len(state["cell_ids"]) else None
        result["last_fit"] = state["fit"]
        # cleanup registry
        try:
            for k, tup in enumerate(list(_LIVE_WIDGETS)):
                if tup[0] is fig:
                    _LIVE_WIDGETS.pop(k)
                    break
        except Exception:
            pass
    fig.canvas.mpl_connect("close_event", _on_close)

    # Nice keyboard shortcuts: s=Save, n=Next, q=Close
    def _on_key(evt):
        if evt.key is None: 
            return
        key = evt.key.lower()
        if key == "s":
            _on_save(None)
        elif key in ("n", "right"):
            _on_next(None)
        elif key in ("q", "escape"):
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show(block=block)
    return result

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def overlay_aligned_at_fit_end(
    df_all,
    summary_df,
    cell_ids=None,                 # if None, use all rows in summary_df with a valid 'end'
    value_col="pattern_score_norm",
    time_col="time_point",
    show_fit_windows=True,         # draw each fitted line segment on top
    individual_alpha=0.3,
    linewidth=1.0,
    title="All raw trajectories aligned to fit end",
):
    """
    Overlay raw trajectories with time shifted so that each cell's fitted-window
    END is at t=0. No truncation to common length; each trace keeps its full span.

    Parameters
    ----------
    df_all : DataFrame with columns [cell_id, time_col, value_col]
    summary_df : DataFrame with columns [cell_id, start, end, length, slope, intercept]
    cell_ids : list[int] or None
    show_fit_windows : bool
        If True, overlay each cell's fitted line segment over its aligned time.
    individual_alpha : float
        Alpha for raw trajectories.
    linewidth : float
        Line width for raw trajectories.
    title : str

    Returns
    -------
    dict with:
        - 'used_cells': list of cell_ids actually plotted
        - 'skipped_cells': list of cell_ids skipped due to missing fit or out-of-range indices
    """
    # pick cells
    if cell_ids is None:
        rows = summary_df.dropna(subset=["cell_id", "end"]).copy()
        sel = rows["cell_id"].astype(int).tolist()
    else:
        sel = list(cell_ids)

    rows_by_id = summary_df.set_index("cell_id")
    used, skipped = [], []

    plt.figure()
    ax = plt.gca()

    for cid in sel:
        if cid not in rows_by_id.index:
            skipped.append(cid); continue

        r = rows_by_id.loc[cid]
        if not np.isfinite(r.get("end", np.nan)):
            skipped.append(cid); continue

        end = int(r["end"])
        start = int(r["start"]) if np.isfinite(r.get("start", np.nan)) else None
        length = int(r["length"]) if np.isfinite(r.get("length", np.nan)) else None
        slope = float(r["slope"]) if np.isfinite(r.get("slope", np.nan)) else None
        intercept = float(r["intercept"]) if np.isfinite(r.get("intercept", np.nan)) else None

        # raw trajectory
        y = _trajectory_from_df(df_all, cid, value_col=value_col, time_col=time_col)
        if y.size == 0 or end < 0 or end >= len(y):
            skipped.append(cid); continue

        # align entire series by shifting x so that end -> t_rel=0
        t_rel = np.arange(len(y), dtype=int) - end
        ax.plot(t_rel, y, alpha=individual_alpha, linewidth=linewidth)

        # optional: draw the fitted line segment in aligned coordinates
        if show_fit_windows and (start is not None) and (length is not None) and (slope is not None) and (intercept is not None):
            # window indices in original time: start..end  (length = end-start+1)
            n = end - start + 1
            if n == length and start >= 0 and end < len(y):
                xw_rel = np.arange(-n + 1, 1, dtype=float)  # since end aligns to 0
                yw_fit = slope * np.arange(n, dtype=float) + intercept
                ax.plot(xw_rel, yw_fit, linewidth=2)  # thicker so it stands out

        used.append(cid)

    ax.axvline(0, linestyle="--", alpha=0.6)  # alignment marker
    ax.set_title(f"{title} (N={len(used)})")
    ax.set_xlabel("time relative to fit end (index)")
    ax.set_ylabel(value_col)
    plt.show()

    return {"used_cells": used, "skipped_cells": skipped}

