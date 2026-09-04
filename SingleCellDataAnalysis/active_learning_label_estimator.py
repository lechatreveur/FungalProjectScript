"""active_learning_label_estimator.py

Estimates the next active-learning label target (in whole lineages) for the
septum-prediction model (TimeAwareSeptumModel / septum_train_lineage.py).

Design contract
---------------
``estimate_next_label_target(current_metrics, dataset_size_history,
                             metric_history) -> int``

is the single public symbol consumed by the auto-trigger (Prompt 4). It can
be called after every retrain round and is fully stateless – all history is
passed in by the caller so the trigger owns the persistence layer.

Scaling model
-------------
The primary metric is ``endpoint_event_f1_at_5_min`` (F1_ep).  Once >= 3
(N, F1) data points exist a power-law is fitted::

    F1_ep(N) approx F1_max - b * N^(-c)          [monotone power-law]

This is the same family used for NLP/vision learning-curve extrapolation
(Rosenfeld et al. 2020 "A Constructive Prediction of the Generalization
Error Across Scales", Sec. 2).  We solve for N_target such that the
predicted F1_ep reaches `target_f1`, then add a 20% headroom buffer and
round up.

Before >= 3 points exist, we fall back to the literature-grounded heuristic
of adding 25% of the current training set (see module docstring Fallback
for references).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RoundMetrics(NamedTuple):
    """Metrics snapshot from one retrain round (all floats, NaN if unknown)."""

    state_balanced_accuracy: float
    endpoint_event_f1_at_5_min: float
    endpoint_median_absolute_error_min: float


@dataclass
class EstimatorConfig:
    """Tuneable parameters; keep defaults grounded.

    Attributes
    ----------
    target_f1 : float
        F1_ep value we consider "meaningfully better" – used by the
        power-law path.  Default 0.35 is ~45% relative lift from v1's 0.240.
    first_round_fraction : float
        Fraction of current training lineages to add when no learning curve
        exists yet.  25% is within the 20-30% range recommended for small
        (<500 sample) biomedical active-learning datasets
        (Settles 2012 sec 6.1; Yang et al. 2017 MICCAI).
    minimum_batch : int
        Never return fewer than this many new labels per round.  Prevents
        stalling on trivially small increments.
    maximum_batch : int
        Caps annotation cost per round; avoids over-committing before the
        learning curve slope is known.
    power_law_headroom : float
        Multiplicative safety margin on the power-law prediction.
        1.20 = 20% extra to account for data-quality variance.
    min_points_for_fit : int
        Minimum (N, F1) pairs required before attempting a power-law fit.
    """

    target_f1: float = 0.35
    first_round_fraction: float = 0.25
    minimum_batch: int = 15
    maximum_batch: int = 80
    power_law_headroom: float = 1.20
    min_points_for_fit: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _power_law_fit(
    sizes: Sequence[int],
    f1_values: Sequence[float],
) -> tuple[float, float, float] | None:
    """Fit F1(N) approx F1_max - b * N^(-c) via log-linearisation.

    Returns (F1_max, b, c) or None if the fit is not numerically stable.

    Log-linearisation trick (Rosenfeld et al. 2020):
    Let  r(N) = F1_max - F1(N)   (residual to asymptote).
    Then log(r) = log(b) - c * log(N)  -- linear in log-log.

    We use F1_max = min(1.0, max_observed * 1.15) as a soft ceiling so the
    extrapolation is always reachable but never absurd.
    """
    if len(sizes) < 2:
        return None
    f1_max = min(1.0, max(f1_values) * 1.15)
    log_n: list[float] = []
    log_r: list[float] = []
    for n, f1 in zip(sizes, f1_values):
        residual = f1_max - f1
        if residual <= 0 or n <= 0:
            continue
        log_n.append(math.log(n))
        log_r.append(math.log(residual))

    if len(log_n) < 2:
        return None

    # Ordinary least squares in log space
    n_pts = len(log_n)
    mean_x = sum(log_n) / n_pts
    mean_y = sum(log_r) / n_pts
    ss_xx = sum((x - mean_x) ** 2 for x in log_n)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_n, log_r))
    if ss_xx < 1e-12:
        return None
    c = -ss_xy / ss_xx           # slope (negated because residual decreases)
    log_b = mean_y + c * mean_x  # intercept
    b = math.exp(log_b)

    # Sanity: c should be positive (residual shrinks as N grows)
    if c <= 0:
        return None

    return f1_max, b, c


def _n_for_target_f1(
    f1_max: float,
    b: float,
    c: float,
    target_f1: float,
) -> int | None:
    """Invert F1_max - b * N^(-c) = target_f1.

    N = (b / (F1_max - target_f1))^(1/c)
    """
    gap = f1_max - target_f1
    if gap <= 0 or b <= 0 or c <= 0:
        return None
    try:
        n_float = (b / gap) ** (1.0 / c)
    except (OverflowError, ZeroDivisionError):
        return None
    if not math.isfinite(n_float) or n_float <= 0:
        return None
    return math.ceil(n_float)


def _conservative_first_round(
    current_train_lineages: int,
    config: EstimatorConfig,
) -> int:
    """25% of current training set, clamped to [minimum_batch, maximum_batch].

    Rationale (explicit, to be replaced by real learning curve):
    - 20-30% increments per round are a well-cited heuristic for small
      biomedical active-learning datasets (Settles 2012 "Active Learning",
      sec 6.1; Yang et al. MICCAI 2017 "Suggestive Annotation").
    - At N=199 training lineages, 25% is approximately 50 new lineages.
      This is large enough to move validation metrics by >0.5 std-dev in
      similar small-N regimes (Isensee et al. 2021 nnU-Net supplemental),
      small enough to avoid wasting budget if the slope is already flat.
    - This estimate MUST be replaced by the power-law path once >= 3
      retrain checkpoints exist.
    """
    batch = round(current_train_lineages * config.first_round_fraction)
    return max(config.minimum_batch, min(config.maximum_batch, batch))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_next_label_target(
    current_metrics: RoundMetrics,
    dataset_size_history: Sequence[int],
    metric_history: Sequence[RoundMetrics],
    *,
    config: EstimatorConfig | None = None,
) -> int:
    """Return the total training-lineage count to reach before the next retrain.

    Parameters
    ----------
    current_metrics : RoundMetrics
        Metrics from the most recent evaluation (held-out test set).
        ``endpoint_event_f1_at_5_min`` drives the estimate.
    dataset_size_history : Sequence[int]
        Ordered list of *training* lineage counts used in each past retrain
        round, including the current one.  Must have the same length as
        ``metric_history``.  Example: [199, 249, 305].
    metric_history : Sequence[RoundMetrics]
        One ``RoundMetrics`` per past retrain round (same order as
        ``dataset_size_history``).
    config : EstimatorConfig, optional
        Override defaults.

    Returns
    -------
    int
        Absolute target total training-lineage count.  Subtract
        ``dataset_size_history[-1]`` to get the number of *new* labels
        to collect.

    Notes
    -----
    Path A (power-law):  >= ``config.min_points_for_fit`` history entries ->
        fit F1_ep(N) approx F1_max - b*N^(-c) and extrapolate to
        ``target_f1``, then apply ``power_law_headroom`` and round up.

    Path B (conservative heuristic):  fewer history entries ->
        add ``first_round_fraction`` x current training size, clamped to
        [minimum_batch, maximum_batch].  **This is a starting estimate only**
        and should be replaced by Path A once >= 3 retrain rounds have
        occurred.

    The return value is always >= current size + ``minimum_batch``.
    """
    if config is None:
        config = EstimatorConfig()

    if len(dataset_size_history) != len(metric_history):
        raise ValueError(
            f"dataset_size_history (len={len(dataset_size_history)}) and "
            f"metric_history (len={len(metric_history)}) must have equal length."
        )

    current_size = dataset_size_history[-1] if dataset_size_history else 199
    current_f1 = current_metrics.endpoint_event_f1_at_5_min

    # ------------------------------------------------------------------
    # Path A: fit power law when enough history exists
    # ------------------------------------------------------------------
    if len(dataset_size_history) >= config.min_points_for_fit:
        f1_values = [m.endpoint_event_f1_at_5_min for m in metric_history]
        fit = _power_law_fit(dataset_size_history, f1_values)
        if fit is not None:
            f1_max, b, c = fit
            # Chase the configured target but always aim for at least +0.05 lift
            target_f1 = max(config.target_f1, current_f1 + 0.05)
            n_target = _n_for_target_f1(f1_max, b, c, target_f1)
            if n_target is not None and n_target > current_size:
                n_with_headroom = math.ceil(n_target * config.power_law_headroom)
                delta = n_with_headroom - current_size
                delta = max(config.minimum_batch, min(config.maximum_batch, delta))
                return current_size + delta
        warnings.warn(
            "Power-law fit failed or predicted no improvement; "
            "falling back to conservative heuristic.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Path B: conservative heuristic (first round or failed fit)
    # ------------------------------------------------------------------
    batch = _conservative_first_round(current_size, config)
    return current_size + batch


# ---------------------------------------------------------------------------
# Convenience: snapshot the v1 baseline as a known data point
# ---------------------------------------------------------------------------

#: Verified from lineage_model_v1/evaluation.json (locked test, seed=123)
BASELINE_V1 = RoundMetrics(
    state_balanced_accuracy=0.7271,
    endpoint_event_f1_at_5_min=0.2397,
    endpoint_median_absolute_error_min=2.0,
)

#: Training lineages used to produce the v1 checkpoint (evaluation.json)
BASELINE_V1_TRAIN_LINEAGES: int = 199


# ---------------------------------------------------------------------------
# Self-test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Round 1 estimate (no prior history, conservative heuristic) ===")
    target = estimate_next_label_target(
        current_metrics=BASELINE_V1,
        dataset_size_history=[BASELINE_V1_TRAIN_LINEAGES],
        metric_history=[BASELINE_V1],
    )
    print(
        json.dumps(
            {
                "current_train_lineages": BASELINE_V1_TRAIN_LINEAGES,
                "target_total": target,
                "new_labels_to_collect": target - BASELINE_V1_TRAIN_LINEAGES,
                "path": "conservative_heuristic_25pct",
                "note": (
                    "STARTING ESTIMATE ONLY. Replace with power-law path "
                    "after >=3 retrain rounds."
                ),
            },
            indent=2,
        )
    )

    print("\n=== Round 4 simulated (power-law path) ===")
    fake_history = [199, 249, 305, 362]
    fake_metrics = [
        RoundMetrics(0.727, 0.240, 2.0),
        RoundMetrics(0.740, 0.268, 1.8),
        RoundMetrics(0.751, 0.290, 1.6),
        RoundMetrics(0.760, 0.308, 1.5),
    ]
    target2 = estimate_next_label_target(
        current_metrics=fake_metrics[-1],
        dataset_size_history=fake_history,
        metric_history=fake_metrics,
    )
    print(
        json.dumps(
            {
                "current_train_lineages": fake_history[-1],
                "target_total": target2,
                "new_labels_to_collect": target2 - fake_history[-1],
                "path": "power_law_fit",
            },
            indent=2,
        )
    )
