"""
Corrects altitude drift in flight telemetry data based on ground phases.

Barometric and GPS altimeters often experience drift over the duration of a flight.
This module anchors the flight track's vertical coordinates (z_m) to the known
ground elevation (0.0m relative to the airfield origin) by analyzing the
PRE_FLIGHT and POST_FLIGHT phases.

The correction is based on the **minimum** z_m observed in each ground phase
(representing the true ground contact altitude). A linearly interpolated
correction is applied across the flight so that the minimum altitude in each
ground phase becomes exactly 0.0m, with constant extrapolation beyond the
anchor points. This avoids hard clamping which would introduce discontinuities.
"""

import logging

import numpy as np
import pandas as pd

from flight_composer.flight_data import Airfield
from flight_composer.processing.assign_flight_phases import FlightPhase

# Threshold (metres) above which a ground-phase offset is suspiciously large
_LARGE_OFFSET_THRESHOLD = 30.0

# Threshold (metres) for high standard deviation within a ground phase
_HIGH_NOISE_THRESHOLD = 5.0

# Number of standard deviations from the mean to flag minimum as an outlier
_OUTLIER_SIGMA = 3.0


def fix_altitude_drift(dataframe: pd.DataFrame, airfield: Airfield) -> pd.DataFrame:
    """
    Corrects z_m altitude drift using pre-flight and post-flight ground phases.

    The function assumes ``z_m`` is already relative to the airfield origin
    (target ground altitude is 0.0 m).  It determines the correction offset at
    each end of the flight from the **minimum** ``z_m`` observed during the
    corresponding ground phase (PRE_FLIGHT / POST_FLIGHT) and shifts the
    entire altitude trace so that these minima become exactly 0.0 m.

    No hard clamping is applied — the correction is smooth (linear
    interpolation between the two anchor points, constant beyond them), which
    avoids introducing altitude discontinuities at phase boundaries.

    Outlier detection compares each phase's minimum ``z_m`` against the mean
    computed **only** over PRE_FLIGHT and POST_FLIGHT samples (LANDING_ROLL
    is excluded because it is sometimes classified too early).

    Correction logic
    ----------------
    1. **Two ground phases** (PRE_FLIGHT *and* POST_FLIGHT present):
       Computes the minimum ``z_m`` in each phase.  These two offsets are
       linearly interpolated across the flight (anchored at the median
       timestamp of each phase) and subtracted from ``z_m``.  Beyond the
       anchor points the correction is held constant, so the minimum
       altitude in each ground phase becomes 0.0 m.

    2. **Single ground phase** (only one of PRE_FLIGHT / POST_FLIGHT):
       Subtracts the minimum ``z_m`` of that phase as a constant offset
       from the entire ``z_m`` column, so the lowest ground-phase reading
       becomes 0.0 m.

    3. **No ground phases**: Returns the dataframe unmodified.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Flight telemetry dataframe containing ``timestamp_s``, ``z_m``, and
        ``phase_hmm`` columns.
    airfield : Airfield
        The airfield associated with the flight track (used for context and
        future origin validation).

    Returns
    -------
    pd.DataFrame
        A new dataframe with the corrected ``z_m`` coordinates.
    """
    logger = logging.getLogger("flight_track.fix_altitude_drift")

    df = dataframe.copy()

    # ------------------------------------------------------------------
    # Identify ground phases
    # ------------------------------------------------------------------
    pre_mask = df["phase_hmm"] == FlightPhase.PRE_FLIGHT
    post_mask = df["phase_hmm"] == FlightPhase.POST_FLIGHT

    has_pre = pre_mask.any()
    has_post = post_mask.any()

    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())

    logger.info(f"Ground phase samples: PRE_FLIGHT={n_pre}, POST_FLIGHT={n_post}")

    # ------------------------------------------------------------------
    # No ground phases — nothing we can do
    # ------------------------------------------------------------------
    if not has_pre and not has_post:
        logger.warning(
            "No ground phases (PRE_FLIGHT / POST_FLIGHT) detected – "
            "altitude drift correction skipped, returning data unmodified"
        )
        return df

    # ------------------------------------------------------------------
    # Helper: compute stats for a ground phase and check for outliers
    # ------------------------------------------------------------------
    def _phase_stats(
        mask: pd.Series, phase_name: str
    ) -> tuple[float, float, float, float, float]:
        """Return (min_z, mean_z, std_z, median_t, n) and log outlier warnings."""
        z_vals = df.loc[mask, "z_m"]
        t_vals = df.loc[mask, "timestamp_s"]

        min_z = float(z_vals.min())
        mean_z = float(z_vals.mean())
        std_z = float(z_vals.std()) if len(z_vals) > 1 else 0.0
        median_t = float(t_vals.median())
        n = int(mask.sum())

        logger.info(
            f"{phase_name} stats: min z_m={min_z:+.2f}m, mean z_m={mean_z:+.2f}m, "
            f"std={std_z:.2f}m, median t={median_t:.1f}s (n={n})"
        )

        # Outlier check: is the minimum far from the phase mean?
        if std_z > 0 and abs(min_z - mean_z) > _OUTLIER_SIGMA * std_z:
            logger.warning(
                f"Anomaly: {phase_name} minimum z_m={min_z:+.2f}m is "
                f"{abs(min_z - mean_z) / std_z:.1f}σ from the phase mean "
                f"({mean_z:+.2f}m) – possible outlier / GPS glitch"
            )

        # Large offset warning
        if abs(min_z) > _LARGE_OFFSET_THRESHOLD:
            logger.warning(
                f"Anomaly: {phase_name} minimum z_m={min_z:+.2f}m exceeds "
                f"±{_LARGE_OFFSET_THRESHOLD}m – possible altimeter or origin error"
            )

        # High noise warning
        if std_z > _HIGH_NOISE_THRESHOLD:
            logger.warning(
                f"Anomaly: {phase_name} z_m noise is high (std={std_z:.2f}m) – "
                f"ground altitude estimate may be unreliable"
            )

        return min_z, mean_z, std_z, median_t, n

    # ------------------------------------------------------------------
    # Both ground phases → linear interpolation
    # ------------------------------------------------------------------
    if has_pre and has_post:
        logger.info(
            "Correction mode: linear interpolation (both ground phases present)"
        )

        pre_min_z, pre_mean_z, pre_std_z, pre_median_t, _ = _phase_stats(
            pre_mask, "PRE_FLIGHT"
        )
        post_min_z, post_mean_z, post_std_z, post_median_t, _ = _phase_stats(
            post_mask, "POST_FLIGHT"
        )

        total_drift = post_min_z - pre_min_z
        logger.info(
            f"Total altitude drift over flight: {total_drift:+.2f}m "
            f"(pre min={pre_min_z:+.2f}m → post min={post_min_z:+.2f}m)"
        )

        if pre_median_t >= post_median_t:
            logger.warning(
                f"Anomaly: PRE_FLIGHT median timestamp ({pre_median_t:.1f}s) is not "
                f"before POST_FLIGHT ({post_median_t:.1f}s) – phase ordering may be wrong"
            )

        # Linearly interpolate the correction between the two anchor points,
        # held constant (flat) beyond the anchors.
        correction = np.interp(
            np.asarray(df["timestamp_s"], dtype=np.float64),
            [pre_median_t, post_median_t],
            [pre_min_z, post_min_z],
        )

        df["z_m"] = df["z_m"] - correction

        logger.info(
            f"Applied linear correction: {pre_min_z:+.2f}m at t={pre_median_t:.1f}s "
            f"→ {post_min_z:+.2f}m at t={post_median_t:.1f}s"
        )

    # ------------------------------------------------------------------
    # Single ground phase → constant offset
    # ------------------------------------------------------------------
    else:
        if has_pre:
            phase_name = "PRE_FLIGHT"
            min_z, _, _, _, _ = _phase_stats(pre_mask, phase_name)
        else:
            phase_name = "POST_FLIGHT"
            min_z, _, _, _, _ = _phase_stats(post_mask, phase_name)

        missing_phase = "POST_FLIGHT" if has_pre else "PRE_FLIGHT"
        logger.info(f"Correction mode: constant offset (only {phase_name} present)")
        logger.warning(
            f"Only one ground phase available ({phase_name}); {missing_phase} is missing – "
            f"cannot estimate drift rate, applying flat correction of {min_z:+.2f}m"
        )

        df["z_m"] = df["z_m"] - min_z

        logger.info(f"Applied constant correction: {min_z:+.2f}m across entire flight")

    return df
