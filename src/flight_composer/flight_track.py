"""
FlightTrack holds discrete series of flight data points before smoothing interpolation is applied.

FlightTrack imports data from GoProTelemetry or IGCTelemetry object. Some preliminary data processing is done by the class.

1. GoPro data sanitization: sometimes GoPro hangs and records invalid timestamps, e.g.

>>> d = extract_gopro_data(pathlib.Path("/srv/samba/share/GoProFlights/09_termika_maj_2025.MP4"))
>>> print(d.data["timestamp_s"][15880:15960].diff().to_string())

2. IGC data sanitization: sometimes IGC has 0 time delta between records. Also, data recorded with my phone often loses altitude reading and suddenly drops to the ground, e.g.

>>> from flight_composer.processing.extract_igc_data import extract_igc_data
>>> d = extract_igc_data(pathlib.Path("IGCData/07_niskie_ladowanie.igc"))
>>> d.data[:][133:142]
     timestamp_s  gps_lat_deg  gps_lon_deg  alt_baro_m  alt_gps_m fix_validity
133        134.0    52.279633    20.908900           0        382            A
134        135.0    52.279700    20.909250           0        140            A
135        136.0    52.279817    20.909550           0        140            A
136        137.0    52.279950    20.909800           0        140            A
137        138.0    52.280100    20.909967           0        140            A
138        139.0    52.280283    20.910067           0        140            A
139        140.0    52.280483    20.910083           0        140            A
140        141.0    52.280683    20.910017           0        140            A
141        142.0    52.280867    20.909867           0        372            A

3. Detect flight phases and add them to the dataframe (column "phase").

4. Fix altitude drift and level ground phases to Z = 0.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa
import pyproj

from flight_composer.flight_data import GoProTelemetry, IGCTelemetry, WGS84Coordinate
from flight_composer.projection import DEFAULT_ORIGIN

# ---------------------------------------------------------------------------
# FlightTrack dataframe schema
# ---------------------------------------------------------------------------

flight_track_schema = pa.DataFrameSchema(
    columns={
        "timestamp_s": pa.Column(
            float,
            checks=pa.Check(
                lambda s: s.is_monotonic_increasing,
                name="is_monotonic",
                error="timestamp_s must be monotonically increasing. Did sanitization/sorting fail?",
            ),
            nullable=False,
        ),
        "x_m": pa.Column(float, nullable=False),
        "y_m": pa.Column(float, nullable=False),
        "z_m": pa.Column(float, nullable=False),
    },
    # Set strict=False so we don't throw errors if 'gps_speed2d_ms' or 'weight' are also present
    strict=False,
    # Coerce ints to floats automatically if needed
    coerce=True,
)

# ---------------------------------------------------------------------------
# IGC data sanitization
# ---------------------------------------------------------------------------


def sanitize_igc_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where timestamp_s does not strictly increase, and fix altitude anomalies.

    Phone-based IGC loggers suffer from two main issues:
    1. Dropout gaps: 3D fix is lost mid-flight, altitude drops to ground level, then recovers.
    2. Missed climbs: 3D fix is lost on the ground, glider launches, and GPS suddenly
       wakes up at 400m+, creating a massive vertical wall.

    This function detects physically impossible vertical velocity jumps (> 30 m/s).
    It pairs drops with recoveries for standard linear interpolation, and it fixes
    missed climbs by reconstructing a typical 10 m/s launch backward from the jump.
    """
    logger = logging.getLogger("flight_track.sanitize_igc_telemetry")
    original_count = len(df)

    # --- PASS 1: Monotonicity ---
    cummax = df["timestamp_s"].cummax().shift(1)
    monotonic_mask = df["timestamp_s"] > cummax
    monotonic_mask.iloc[0] = True

    dropped = original_count - monotonic_mask.sum()
    if dropped > 0:
        logger.info(
            f"IGC sanitization: dropped {dropped}/{original_count} rows "
            f"with non-strictly-increasing timestamps."
        )

    df = df.loc[monotonic_mask, :].copy().reset_index(drop=True)

    # --- PASS 2: Interpolate altitude glitches ---
    MAX_VZ_GLITCH = 30.0
    ASSUMED_CLIMB_RATE = 10.0  # m/s for backwards reconstruction of missed launches

    alt_col = "alt_gps_m"
    if alt_col not in df.columns or len(df) < 3:
        return df

    t = df["timestamp_s"].values
    alt = df[alt_col].values.astype(np.float64)

    # Calculate step-to-step vertical velocity
    vz = np.zeros(len(df), dtype=np.float64)
    dt = np.maximum(np.diff(t), 1.0)
    vz[1:] = np.diff(alt) / dt

    anomaly_indices = np.where(np.abs(vz) > MAX_VZ_GLITCH)[0]
    interp_cols = [c for c in ("alt_gps_m", "alt_baro_m", "z_m") if c in df.columns]

    segments_fixed = 0
    climbs_reconstructed = 0

    i = 0
    while i < len(anomaly_indices):
        idx = anomaly_indices[i]

        # Case 1: Massive drop (Dropout gap)
        if vz[idx] < -MAX_VZ_GLITCH:
            recovery_idx = None
            # Scan forward to find the positive recovery jump
            for j in range(i + 1, len(anomaly_indices)):
                if vz[anomaly_indices[j]] > MAX_VZ_GLITCH:
                    recovery_idx = anomaly_indices[j]
                    i = j  # Consume this recovery so we don't process it twice
                    break

            if recovery_idx is not None:
                anchor_before = idx - 1
                if anchor_before >= 0:
                    bad_indices = np.arange(idx, recovery_idx)
                    t_before, t_after = t[anchor_before], t[recovery_idx]
                    frac = (t[bad_indices] - t_before) / (t_after - t_before)

                    for col in interp_cols:
                        val_before = df.at[anchor_before, col]
                        val_after = df.at[recovery_idx, col]
                        interpolated = val_before + frac * (val_after - val_before)
                        df.loc[bad_indices, col] = interpolated.astype(df[col].dtype)

                    segments_fixed += 1
                    logger.info(
                        f"IGC sanitization: patched dropout gap "
                        f"t=[{t_before:.1f}s .. {t_after:.1f}s], ({len(bad_indices)} points)."
                    )
            i += 1

        # Case 2: Massive jump UP (Missed climb / Delayed fix)
        elif vz[idx] > MAX_VZ_GLITCH:
            anchor_before = idx - 1
            if anchor_before >= 0:
                target_alt = df.at[idx, alt_col]
                ground_alt = df.at[anchor_before, alt_col]
                dz = target_alt - ground_alt

                # Calculate how long the launch theoretically took
                dt_climb = max(1.0, dz / ASSUMED_CLIMB_RATE)
                t_target = t[idx]
                t_start = t_target - dt_climb

                # Find all records that fall into our reconstructed climb window
                climb_mask = (t < t_target) & (t >= t_start)

                if np.any(climb_mask):
                    for col in interp_cols:
                        val_target = df.at[idx, col]
                        val_ground = df.at[anchor_before, col]
                        col_rate = (val_target - val_ground) / dt_climb

                        # Apply the synthesized climb ramp
                        new_vals = val_ground + (t[climb_mask] - t_start) * col_rate
                        df.loc[climb_mask, col] = new_vals.astype(df[col].dtype)

                    climbs_reconstructed += 1
                    logger.info(
                        f"IGC sanitization: reconstructed missed climb "
                        f"t=[{t_start:.1f}s .. {t_target:.1f}s], ({climb_mask.sum()} points)."
                    )
            i += 1
        else:
            i += 1

    if segments_fixed > 0 or climbs_reconstructed > 0:
        logger.info(
            f"IGC sanitization complete: patched {segments_fixed} gaps, "
            f"reconstructed {climbs_reconstructed} missed climbs."
        )

    return df


# ---------------------------------------------------------------------------
# GoPro data sanitization
# ---------------------------------------------------------------------------


def sanitize_gopro_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("flight_track.sanitize_gopro_telemetry")

    original_count = len(df)

    # --- PASS 1: Monotonicity ---
    cummax = df["timestamp_s"].cummax().shift(1)
    monotonic_mask = df["timestamp_s"] > cummax
    monotonic_mask.iloc[0] = True
    df_clean = df.loc[monotonic_mask, :].copy().reset_index(drop=True)

    # --- PASS 2: Kinematics ---
    dist = np.sqrt(df_clean["x_m"].diff() ** 2 + df_clean["y_m"].diff() ** 2)
    dt = df_clean["timestamp_s"].diff()
    implied_speed = dist / dt.replace(0, np.nan)
    hardware_speed = df_clean["gps_speed2d_ms"]

    speed_error_ratio = implied_speed / hardware_speed.replace(0, np.nan)

    # Thresholds
    # Higher tolerance for ground handling
    GROUND_SPEED_THRESHOLD = 7.0  # m/s (~25 km/h)

    valid_kinematics = (speed_error_ratio > 0.5) & (speed_error_ratio < 2.0)
    valid_hardware_rate = dt > 0.03
    is_on_ground = hardware_speed < GROUND_SPEED_THRESHOLD

    # Final mask: Keep if kinematics are valid OR if we are just moving slowly on ground
    valid_mask = (valid_kinematics & valid_hardware_rate) | is_on_ground
    valid_mask.iloc[0] = True

    # --- DIAGNOSTICS ---
    dropped_indices = df_clean.index[~valid_mask]
    num_dropped = len(dropped_indices)

    if num_dropped > 0:
        ground_drops = (
            df_clean.loc[dropped_indices, "gps_speed2d_ms"] < GROUND_SPEED_THRESHOLD
        ).sum()
        flight_drops = num_dropped - ground_drops

        # Identify the time windows of the drops
        ranges = []
        if len(dropped_indices) > 0:
            start_idx = dropped_indices[0]
            for i in range(1, len(dropped_indices)):
                if dropped_indices[i] != dropped_indices[i - 1] + 1:
                    t_start = df_clean.loc[start_idx, "timestamp_s"]
                    t_end = df_clean.loc[dropped_indices[i - 1], "timestamp_s"]
                    ranges.append((t_start, t_end))
                    start_idx = dropped_indices[i]
            # Add final range
            ranges.append(
                (
                    df_clean.loc[start_idx, "timestamp_s"],
                    df_clean.loc[dropped_indices[-1], "timestamp_s"],
                )
            )

        # Filter for "Significant Gaps" (> 1.0 second)
        significant_gaps = [(s, e, e - s) for s, e in ranges if (e - s) > 1.0]
        jitter_count = len(ranges) - len(significant_gaps)

        logger.warning(
            f"Sanitization Report ({original_count} total pts):\n"
            f"  - Dropped {num_dropped} points ({ground_drops} ground, {flight_drops} flight).\n"
            f"  - Isolated jitter events (< 1s): {jitter_count}"
        )

        if significant_gaps:
            gap_str = ", ".join(
                [
                    f"[{s:.1f}s to {e:.1f}s, dur: {d:.2f}s]"
                    for s, e, d in significant_gaps
                ]
            )
            logger.error(f"  CRITICAL GAPS DETECTED (>1s): {gap_str}")
        else:
            logger.info(
                "  No significant gaps detected (>1s). Spline interpolation will be highly accurate."
            )

    return df_clean[valid_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flight phase classification
# ---------------------------------------------------------------------------


class FlightPhase(StrEnum):
    PRE_FLIGHT = "PRE_FLIGHT"
    LAUNCH_ROLL = "LAUNCH_ROLL"
    LAUNCH_CLIMB = "LAUNCH_CLIMB"
    CRUISING = "CRUISING"
    THERMALLING = "THERMALLING"
    LANDING_ROLL = "ROLLING_LANDING"
    POST_FLIGHT = "PUSHING_GROUND"


def assign_flight_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a flight phase to each row in a glider telemetry dataframe using a
    bidirectional, debounced state machine.

    Unlike a strict monotonic system, this algorithm allows transitions back to
    flight states if false landings are detected. Transitions require sustained
    evidence (debouncing) to reject GPS noise and momentary data spikes.

    Expected Input:
        A pandas DataFrame with monotonic 'timestamp_s', 'x_m', 'y_m', 'z_m'.

    Preprocessing & Kinematics:
        - z_agl (m): Altitude above ground level. Ground is the median of the first 10s.
        - v_2d (m/s): Horizontal ground speed (3s central finite difference).
        - v_z (m/s): Vertical speed (3s central finite difference).
        - omega (deg/s): Turn rate, calculated via unwrapped heading delta.

        *Note on smoothing: All kinematic arrays are smoothed with a 2-second
        rolling median to reject quantization noise before state evaluation.*

    State Machine Logic:
        Transitions evaluate sequentially. If conditions for a transition are met
        for 'N' consecutive seconds (the debounce window), the state updates.

        | Current State | Next State | Trigger Conditions | Debounce |
        | :--- | :--- | :--- | :--- |
        | PRE_FLIGHT | LAUNCH_ROLL | v_2d > 3.0 m/s | 1 s |
        | LAUNCH_ROLL | LAUNCH_CLIMB | v_2d > 15.0 m/s AND v_z > 1.0 m/s AND z_agl > 5.0 m | 2 s |
        | LAUNCH_CLIMB | CRUISING | v_z < 1.0 m/s AND z_agl > 50.0 m (Cable Release) | 3 s |
        | CRUISING | THERMALLING | |omega| > 6.0 deg/s | 5 s |
        | THERMALLING | CRUISING | |omega| < 4.0 deg/s | 5 s |
        | (Any Flight) | LANDING_ROLL | z_agl < 10.0 m AND v_2d < 30.0 m/s | 4 s |
        | LANDING_ROLL | POST_FLIGHT | v_2d < 2.0 m/s | 5 s |

    Error Recovery (The "Oops" Rules):
        - If in LANDING_ROLL or POST_FLIGHT, but z_agl > 30.0 m AND v_2d > 15.0 m/s,
            immediately revert to CRUISING (recovers from false ground-level GPS spikes).

    Returns:
        pd.DataFrame: The dataframe with an appended 'phase' column.
    """
    logger = logging.getLogger("flight_track.assign_flight_phases")

    validated_df = flight_track_schema.validate(df).copy()

    n = len(validated_df)
    if n == 0:
        validated_df["phase"] = pd.Series(dtype=str)
        return validated_df

    t = validated_df["timestamp_s"].values.astype(np.float64)
    x = validated_df["x_m"].values.astype(np.float64)
    y = validated_df["y_m"].values.astype(np.float64)
    z = validated_df["z_m"].values.astype(np.float64)

    # -----------------------------------------------------------------------
    # Preprocessing: Compute kinematics
    # -----------------------------------------------------------------------

    # z_agl: altitude above ground level (ground = median of first 10 seconds)
    first_10s_mask = t <= t[0] + 10.0
    n_ground = int(np.sum(first_10s_mask))
    if n_ground < 1:
        n_ground = 1
    ground_z = float(np.median(z[:n_ground]))
    z_agl = z - ground_z

    logger.info(
        f"Ground level z={ground_z:.1f} m (median of first {n_ground} samples / 10s)"
    )

    # Central finite differences over a 3-second window.
    # For each point i, find the nearest indices ~1.5s before and ~1.5s after.
    DIFF_HALF_WINDOW = 1.5  # seconds

    # Pre-compute index offsets: for each row, the backward and forward partner
    idx_back = np.zeros(n, dtype=int)
    idx_fwd = np.zeros(n, dtype=int)
    j_back = 0
    j_fwd = 0
    for i in range(n):
        # Advance j_back so that t[i] - t[j_back] is closest to DIFF_HALF_WINDOW
        while j_back < i - 1 and t[i] - t[j_back + 1] >= DIFF_HALF_WINDOW:
            j_back += 1
        idx_back[i] = min(j_back, i)

    j_fwd = n - 1
    for i in range(n - 1, -1, -1):
        while j_fwd > i + 1 and t[j_fwd - 1] - t[i] >= DIFF_HALF_WINDOW:
            j_fwd -= 1
        idx_fwd[i] = max(j_fwd, i)

    # Ensure we don't divide by zero; clamp minimum dt to a tiny value
    dt_diff = t[idx_fwd] - t[idx_back]
    dt_diff = np.where(dt_diff < 1e-6, 1e-6, dt_diff)

    dx = x[idx_fwd] - x[idx_back]
    dy = y[idx_fwd] - y[idx_back]
    dz = z[idx_fwd] - z[idx_back]

    v_2d_raw = np.sqrt(dx**2 + dy**2) / dt_diff
    v_z_raw = dz / dt_diff

    # Heading from velocity direction (radians)
    heading_raw = np.arctan2(dy, dx)
    # Unwrap to remove ±π discontinuities
    heading_unwrapped = np.unwrap(heading_raw)
    # Turn rate: d(heading)/dt via same central difference partners
    d_heading = heading_unwrapped[idx_fwd] - heading_unwrapped[idx_back]
    omega_raw = np.degrees(d_heading / dt_diff)  # deg/s

    # NOISE GATE: If horizontal speed is less than 2.0 m/s (walking pace),
    # the heading is just GPS jitter. Force turn rate to 0.
    omega_raw[v_2d_raw < 2.0] = 0.0

    # -----------------------------------------------------------------------
    # Smoothing: 2-second rolling median
    # -----------------------------------------------------------------------
    # Determine the rolling window size in samples from the median sample rate.
    if n > 1:
        median_dt = float(np.median(np.diff(t)))
        if median_dt < 1e-6:
            median_dt = 1.0
        smooth_window = max(3, int(round(2.0 / median_dt)))
        # Ensure odd for centered rolling median
        if smooth_window % 2 == 0:
            smooth_window += 1
    else:
        smooth_window = 3

    def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
        s = pd.Series(arr)
        return np.asarray(s.rolling(win, center=True, min_periods=1).median().values)

    v_2d = _rolling_median(v_2d_raw, smooth_window)
    v_z = _rolling_median(v_z_raw, smooth_window)
    omega = _rolling_median(omega_raw, smooth_window)

    # -----------------------------------------------------------------------
    # Log anomalies in the kinematic data
    # -----------------------------------------------------------------------
    max_v2d = float(np.nanmax(v_2d))
    max_vz = float(np.nanmax(np.abs(v_z)))
    max_omega = float(np.nanmax(np.abs(omega)))
    logger.info(
        f"Kinematics summary: max v_2d={max_v2d:.1f} m/s, "
        f"max |v_z|={max_vz:.1f} m/s, max |omega|={max_omega:.1f} deg/s"
    )

    # Check for anomalous spikes
    if max_v2d > 80.0:
        logger.warning(
            f"Anomaly: max horizontal speed {max_v2d:.1f} m/s exceeds 80 m/s – "
            f"possible GPS glitch"
        )
    if max_vz > 30.0:
        logger.warning(
            f"Anomaly: max vertical speed {max_vz:.1f} m/s exceeds 30 m/s – "
            f"possible altitude glitch"
        )
    neg_agl_count = int(np.sum(z_agl < -5.0))
    if neg_agl_count > 0:
        logger.warning(
            f"Anomaly: {neg_agl_count} samples with z_agl < -5 m "
            f"(below estimated ground level)"
        )

    # -----------------------------------------------------------------------
    # State machine: debounced, bidirectional
    # -----------------------------------------------------------------------
    phases = [FlightPhase.PRE_FLIGHT] * n
    state = FlightPhase.PRE_FLIGHT

    # Debounce accumulators: accumulated time that each candidate transition
    # condition has been continuously true.
    debounce_acc: dict[str, float] = {
        "launch_roll": 0.0,
        "launch_climb": 0.0,
        "cruising_from_climb": 0.0,
        "thermalling": 0.0,
        "cruising_from_thermal": 0.0,
        "landing_roll": 0.0,
        "post_flight": 0.0,
    }

    def _reset_all_debounce() -> None:
        for k in debounce_acc:
            debounce_acc[k] = 0.0

    prev_state = state

    for i in range(n):
        dt_row = (t[i] - t[i - 1]) if i > 0 else 0.0

        vi_2d = v_2d[i]
        vi_z = v_z[i]
        zi_agl = z_agl[i]
        wi = abs(omega[i])

        # -------------------------------------------------------------------
        # Error recovery: immediate revert from false landing/post-flight
        # -------------------------------------------------------------------
        if state in (FlightPhase.LANDING_ROLL, FlightPhase.POST_FLIGHT):
            if zi_agl > 30.0 and vi_2d > 15.0:
                logger.warning(
                    f"Error recovery at t={t[i]:.1f}s (row {i}): "
                    f"z_agl={zi_agl:.1f}m, v_2d={vi_2d:.1f}m/s – "
                    f"reverting {state} -> CRUISING"
                )
                state = FlightPhase.CRUISING
                _reset_all_debounce()

        # -------------------------------------------------------------------
        # Forward transitions (evaluated in priority order)
        # -------------------------------------------------------------------
        if state == FlightPhase.PRE_FLIGHT:
            if vi_2d > 3.0:
                debounce_acc["launch_roll"] += dt_row
                if debounce_acc["launch_roll"] >= 1.0:
                    state = FlightPhase.LAUNCH_ROLL
                    _reset_all_debounce()
            else:
                debounce_acc["launch_roll"] = 0.0

        elif state == FlightPhase.LAUNCH_ROLL:
            if vi_2d > 15.0 and vi_z > 1.0 and zi_agl > 5.0:
                debounce_acc["launch_climb"] += dt_row
                if debounce_acc["launch_climb"] >= 2.0:
                    state = FlightPhase.LAUNCH_CLIMB
                    _reset_all_debounce()
            else:
                debounce_acc["launch_climb"] = 0.0

        elif state == FlightPhase.LAUNCH_CLIMB:
            # Check landing first (higher priority for safety)
            if zi_agl < 10.0 and vi_2d < 30.0:
                debounce_acc["landing_roll"] += dt_row
                if debounce_acc["landing_roll"] >= 4.0:
                    state = FlightPhase.LANDING_ROLL
                    _reset_all_debounce()
            else:
                debounce_acc["landing_roll"] = 0.0

            # Cable release → cruising
            if state == FlightPhase.LAUNCH_CLIMB:
                if vi_z < 1.0 and zi_agl > 50.0:
                    debounce_acc["cruising_from_climb"] += dt_row
                    if debounce_acc["cruising_from_climb"] >= 3.0:
                        state = FlightPhase.CRUISING
                        _reset_all_debounce()
                else:
                    debounce_acc["cruising_from_climb"] = 0.0

        elif state == FlightPhase.CRUISING:
            # Check landing first
            if zi_agl < 10.0 and vi_2d < 30.0:
                debounce_acc["landing_roll"] += dt_row
                if debounce_acc["landing_roll"] >= 4.0:
                    state = FlightPhase.LANDING_ROLL
                    _reset_all_debounce()
            else:
                debounce_acc["landing_roll"] = 0.0

            # Thermalling detection
            if state == FlightPhase.CRUISING:
                if wi > 6.0:
                    debounce_acc["thermalling"] += dt_row
                    if debounce_acc["thermalling"] >= 5.0:
                        state = FlightPhase.THERMALLING
                        _reset_all_debounce()
                else:
                    debounce_acc["thermalling"] = 0.0

        elif state == FlightPhase.THERMALLING:
            # Check landing first
            if zi_agl < 10.0 and vi_2d < 30.0:
                debounce_acc["landing_roll"] += dt_row
                if debounce_acc["landing_roll"] >= 4.0:
                    state = FlightPhase.LANDING_ROLL
                    _reset_all_debounce()
            else:
                debounce_acc["landing_roll"] = 0.0

            # Exit thermal → cruising
            if state == FlightPhase.THERMALLING:
                if wi < 4.0:
                    debounce_acc["cruising_from_thermal"] += dt_row
                    if debounce_acc["cruising_from_thermal"] >= 5.0:
                        state = FlightPhase.CRUISING
                        _reset_all_debounce()
                else:
                    debounce_acc["cruising_from_thermal"] = 0.0

        elif state == FlightPhase.LANDING_ROLL:
            if vi_2d < 2.0:
                debounce_acc["post_flight"] += dt_row
                if debounce_acc["post_flight"] >= 5.0:
                    state = FlightPhase.POST_FLIGHT
                    _reset_all_debounce()
            else:
                debounce_acc["post_flight"] = 0.0

        # POST_FLIGHT is terminal (unless error recovery fires above)

        phases[i] = state

        # -------------------------------------------------------------------
        # Log phase transitions
        # -------------------------------------------------------------------
        if state != prev_state:
            logger.info(
                f"Phase transition at t={t[i]:.1f}s (row {i}): {prev_state} -> {state}"
            )
            prev_state = state

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    validated_df["phase"] = phases

    phase_counts = validated_df["phase"].value_counts()
    total_duration = t[-1] - t[0] if n > 1 else 0.0
    logger.info(f"Flight duration: {total_duration:.1f}s ({n} samples)")
    for phase_name, count in phase_counts.items():
        phase_mask = validated_df["phase"] == phase_name
        phase_ts = validated_df.loc[phase_mask, "timestamp_s"].values
        if len(phase_ts) > 1:
            phase_dur = float(phase_ts[-1] - phase_ts[0])
        else:
            phase_dur = 0.0
        logger.info(f"  {phase_name}: {count} samples, ~{phase_dur:.1f}s")

    return validated_df


def assign_flight_phases_hmm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns flight phases to glider telemetry using a Hidden Markov Model (HMM)
    and the Viterbi algorithm.

    This approach evaluates the entire flight globally to calculate the
    mathematically most probable sequence of flight phases, naturally filtering
    out GPS noise, momentary dropouts, and altitude anomalies.

    Expected Input:
        df (pd.DataFrame): Telemetry data containing at least:
            - 'timestamp_s' (float): Monotonically increasing time in seconds.
            - 'x_m', 'y_m', 'z_m' (float): Local coordinates in meters.

    Preprocessing & Observables (O):
        - v_2d (m/s): Horizontal ground speed.
        - v_z (m/s): Vertical speed.
        - omega (deg/s): Unwrapped turn rate (yaw rate).
        - z_agl (m): Altitude above ground level.

    Hidden States (S):
        Mapped directly from the FlightPhase StrEnum:
        - PRE_FLIGHT: Stationary or slow taxiing before launch.
        - LAUNCH_ROLL: Accelerating on the ground (winch or aerotow).
        - LAUNCH_CLIMB: Steep or sustained climb connected to the launch.
        - CRUISING: Airborne, relatively straight flight.
        - THERMALLING: Airborne, continuous circling.
        - LANDING_ROLL ("ROLLING_LANDING"): Touchdown and deceleration.
        - POST_FLIGHT ("PUSHING_GROUND"): Stationary or walking pace after landing.

    Mathematical Model:
        1. Transition Matrix (A): Defines the probability of moving from state i
           to state j. It enforces logical flight progression (e.g., PRE_FLIGHT
           cannot transition directly to THERMALLING) while allowing fluid
           back-and-forth between CRUISING and THERMALLING.

        2. Emission Probabilities (B): Modeled as Probability Density Functions
           (PDFs) for each state. For example, the probability of observing
           high v_z and high z_agl is maximized in LAUNCH_CLIMB or THERMALLING,
           but minimized in PRE_FLIGHT.

    Returns:
        pd.DataFrame: A copy of the original dataframe with an appended 'phase_hmm'
        column containing the evaluated FlightPhase string literal.
    """
    logger = logging.getLogger("flight_track.assign_flight_phases_hmm")

    # Assuming flight_track_schema is defined in your broader scope
    validated_df = flight_track_schema.validate(df).copy()

    n = len(validated_df)
    if n == 0:
        validated_df["phase"] = pd.Series(dtype=str)
        return validated_df

    t = validated_df["timestamp_s"].values.astype(np.float64)
    x = validated_df["x_m"].values.astype(np.float64)
    y = validated_df["y_m"].values.astype(np.float64)
    z = validated_df["z_m"].values.astype(np.float64)

    # ===================================================================
    # 1. Kinematic Feature Extraction (vectorized)
    # ===================================================================

    # z_agl: altitude above ground level (ground = median of first 10 s)
    first_10s_mask = t <= t[0] + 10.0
    n_ground = max(1, int(np.sum(first_10s_mask)))
    ground_z = float(np.median(z[:n_ground]))
    z_agl = z - ground_z

    logger.info(
        f"Ground level z={ground_z:.1f} m (median of first {n_ground} samples / 10s)"
    )

    # Central finite differences over a ~3-second window via np.searchsorted
    DIFF_HALF_WINDOW = 1.5  # seconds

    idx_back = np.searchsorted(t, t - DIFF_HALF_WINDOW, side="left")
    idx_fwd = np.searchsorted(t, t + DIFF_HALF_WINDOW, side="right") - 1
    idx_fwd = np.clip(idx_fwd, 0, n - 1)

    # Time span for each pair; clamp to avoid division by zero
    dt_diff = np.maximum(t[idx_fwd] - t[idx_back], 1e-6)

    dx = x[idx_fwd] - x[idx_back]
    dy = y[idx_fwd] - y[idx_back]
    dz = z[idx_fwd] - z[idx_back]

    v_2d_raw = np.sqrt(dx**2 + dy**2) / dt_diff
    v_z_raw = dz / dt_diff

    # Heading from velocity direction (radians), unwrapped to avoid ±π jumps
    heading_raw = np.arctan2(dy, dx)
    heading_unwrapped = np.unwrap(heading_raw)

    # Turn rate via the same central-difference partners
    d_heading = heading_unwrapped[idx_fwd] - heading_unwrapped[idx_back]
    omega_raw = np.degrees(d_heading / dt_diff)  # deg/s

    # NOISE GATE: If horizontal speed is less than 2.0 m/s (walking pace),
    # the heading is just GPS jitter. Force turn rate to 0.
    omega_raw[v_2d_raw < 2.0] = 0.0

    # ---------------------------------------------------------------
    # Light smoothing: ~2-second rolling median
    # ---------------------------------------------------------------
    if n > 1:
        median_dt = float(np.median(np.diff(t)))
        if median_dt < 1e-6:
            median_dt = 1.0
        smooth_window = max(3, int(round(2.0 / median_dt)))
        if smooth_window % 2 == 0:
            smooth_window += 1  # keep odd for centered window
    else:
        smooth_window = 3

    def _rolling_median(arr: np.ndarray, win: int) -> np.ndarray:
        s = pd.Series(arr)
        return s.rolling(win, center=True, min_periods=1).median().to_numpy()

    v_2d = _rolling_median(v_2d_raw, smooth_window)
    v_z = _rolling_median(v_z_raw, smooth_window)
    omega = _rolling_median(omega_raw, smooth_window)
    # z_agl is already smooth (derived from GPS altitude), no extra filter

    # ---------------------------------------------------------------
    # Log kinematic summary & anomalies
    # ---------------------------------------------------------------
    max_v2d = float(np.nanmax(v_2d))
    max_vz = float(np.nanmax(np.abs(v_z)))
    max_omega = float(np.nanmax(np.abs(omega)))
    logger.info(
        f"Kinematics summary: max v_2d={max_v2d:.1f} m/s, "
        f"max |v_z|={max_vz:.1f} m/s, max |omega|={max_omega:.1f} deg/s"
    )
    if max_v2d > 80.0:
        logger.warning(
            f"Anomaly: max horizontal speed {max_v2d:.1f} m/s exceeds 80 m/s – "
            f"possible GPS glitch"
        )
    if max_vz > 30.0:
        logger.warning(
            f"Anomaly: max vertical speed {max_vz:.1f} m/s exceeds 30 m/s – "
            f"possible altitude glitch"
        )
    neg_agl_count = int(np.sum(z_agl < -5.0))
    if neg_agl_count > 0:
        logger.warning(
            f"Anomaly: {neg_agl_count} samples with z_agl < -5 m "
            f"(below estimated ground level)"
        )

    # ===================================================================
    # 2. Hidden States Mapping
    # ===================================================================
    NUM_STATES = 7

    STATE_ENUM = [
        FlightPhase.PRE_FLIGHT,  # 0
        FlightPhase.LAUNCH_ROLL,  # 1
        FlightPhase.LAUNCH_CLIMB,  # 2
        FlightPhase.CRUISING,  # 3
        FlightPhase.THERMALLING,  # 4
        FlightPhase.LANDING_ROLL,  # 5
        FlightPhase.POST_FLIGHT,  # 6
    ]

    # ===================================================================
    # 3. Transition Matrix A  (7×7, then converted to log-space)
    # ===================================================================
    A = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
    SELF = 0.999  # Highly sticky to enforce debouncing

    # PRE_FLIGHT (0): can stay or start rolling
    A[0, 0] = SELF
    A[0, 1] = 1.0 - SELF

    # LAUNCH_ROLL (1): can stay, advance to climb, or abort to landing
    A[1, 1] = SELF
    A[1, 2] = 0.0008
    A[1, 5] = 0.0002

    # LAUNCH_CLIMB (2): can stay, release to cruise, or come down
    A[2, 2] = SELF
    A[2, 3] = 0.0008
    A[2, 5] = 0.0002

    # CRUISING (3): can stay, enter thermal, or land
    A[3, 3] = SELF
    A[3, 4] = 0.0006
    A[3, 5] = 0.0004

    # THERMALLING (4): can stay, straighten out, or land
    A[4, 4] = SELF
    A[4, 3] = 0.0006
    A[4, 5] = 0.0004

    # LANDING_ROLL (5): can stay, stop, or recover to cruise (false landing)
    A[5, 5] = SELF
    A[5, 6] = 0.0008
    A[5, 3] = 0.0002

    # POST_FLIGHT (6): absorbing state (with tiny recovery)
    A[6, 6] = SELF
    A[6, 3] = 1.0 - SELF

    # Normalise each row to sum to 1 (guards against typos)
    A = A / A.sum(axis=1, keepdims=True)

    # Convert to log-space; impossible transitions get log(1e-12) ≈ -27.6
    with np.errstate(divide="ignore"):
        log_A = np.log(A)

    logger.info("Transition matrix A (log-space) built successfully")

    # ===================================================================
    # 4. Emission Probabilities via Gaussian PDFs (vectorized)
    # ===================================================================
    # Note: Airborne states (2, 3, 4) have a flat Gaussian for altitude
    # (mu=800, sigma=1000) so high-altitude flying isn't penalised.
    # fmt: off
    emission_params = {
        #                   v_2d            v_z            z_agl          |omega|
        #               (  mu,  sigma)  (  mu,  sigma)  (  mu,  sigma)  (  mu,  sigma)
        0: {"mu": np.array([  0.5,   0.0,    0.0,   0.0]),
            "sigma": np.array([  5.0,   2.0,    5.0,   5.0])},   # PRE_FLIGHT
        1: {"mu": np.array([ 10.0,   0.5,    1.5,   1.0]),
            "sigma": np.array([  6.0,   1.5,    4.0,   3.0])},   # LAUNCH_ROLL
        2: {"mu": np.array([ 28.0,   5.0,  800.0,   2.0]),
            "sigma": np.array([ 10.0,   3.0, 1000.0,   4.0])},   # LAUNCH_CLIMB
        3: {"mu": np.array([ 25.0,  -0.5,  800.0,   2.0]),
            "sigma": np.array([ 10.0,   2.0, 1000.0,   4.0])},   # CRUISING
        4: {"mu": np.array([ 22.0,   0.5,  800.0,  18.0]),
            "sigma": np.array([ 10.0,   3.0, 1000.0,   8.0])},   # THERMALLING
        5: {"mu": np.array([  8.0,  -0.5,    3.0,   2.0]),
            "sigma": np.array([  8.0,   2.0,   10.0,   4.0])},   # LANDING_ROLL
        6: {"mu": np.array([  0.5,   0.0,    0.0,   0.0]),
            "sigma": np.array([  5.0,   2.0,    5.0,   5.0])},   # POST_FLIGHT
    }
    # fmt: on

    obs = np.column_stack([v_2d, v_z, z_agl, np.abs(omega)])  # (N, 4)
    LOG_SQRT_2PI = np.log(np.sqrt(2.0 * np.pi))
    log_B = np.zeros((n, NUM_STATES), dtype=np.float64)

    for k in range(NUM_STATES):
        mu = emission_params[k]["mu"]  # (4,)
        sigma = emission_params[k]["sigma"]  # (4,)
        z_score = (obs - mu) / sigma  # (N, 4)
        log_p_per_feat = -0.5 * z_score**2 - np.log(sigma) - LOG_SQRT_2PI
        log_B[:, k] = log_p_per_feat.sum(axis=1)

    logger.info("Emission log-probabilities computed for all %d states", NUM_STATES)

    # ===================================================================
    # 5. Viterbi Algorithm (forward pass + backtracking)
    # ===================================================================
    #
    V = np.full((n, NUM_STATES), -np.inf, dtype=np.float64)
    ptr = np.zeros((n, NUM_STATES), dtype=np.intp)

    V[0, 0] = 0.0 + log_B[0, 0]

    # Vectorized Forward pass: eliminates the inner state loop
    for t_idx in range(1, n):
        # path_probs shape is (NUM_STATES, NUM_STATES)
        # Rows = Previous State (j), Cols = Current State (k)
        path_probs = V[t_idx - 1, :, None] + log_A

        # Max path to reach state k from all possible j
        V[t_idx, :] = np.max(path_probs, axis=0) + log_B[t_idx, :]
        ptr[t_idx, :] = np.argmax(path_probs, axis=0)

    # Backtracking
    best_path = np.zeros(n, dtype=np.intp)
    best_path[n - 1] = int(np.argmax(V[n - 1, :]))

    for t_idx in range(n - 2, -1, -1):
        best_path[t_idx] = ptr[t_idx + 1, best_path[t_idx + 1]]

    logger.info("Viterbi decoding complete (%d time steps)", n)

    # ===================================================================
    # 6. Map integer states back to FlightPhase string values
    # ===================================================================
    phases = [STATE_ENUM[s].value for s in best_path]
    validated_df["phase_hmm"] = phases

    # ---------------------------------------------------------------
    # Log phase transitions
    # ---------------------------------------------------------------
    prev_state_idx = best_path[0]
    for i in range(1, n):
        if best_path[i] != prev_state_idx:
            logger.info(
                f"Phase transition at t={t[i]:.1f}s (row {i}): "
                f"{STATE_ENUM[prev_state_idx]} -> {STATE_ENUM[best_path[i]]}"
            )
            prev_state_idx = best_path[i]

    # ---------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------
    phase_counts = validated_df["phase_hmm"].value_counts()
    total_duration = t[-1] - t[0] if n > 1 else 0.0
    logger.info(f"Flight duration: {total_duration:.1f}s ({n} samples)")
    for phase_name, count in phase_counts.items():
        phase_mask = validated_df["phase_hmm"] == phase_name
        phase_ts = validated_df.loc[phase_mask, "timestamp_s"].values
        if len(phase_ts) > 1:
            phase_dur = float(phase_ts[-1] - phase_ts[0])
        else:
            phase_dur = 0.0
        logger.info(f"  {phase_name}: {count} samples, ~{phase_dur:.1f}s")

    return validated_df


# ---------------------------------------------------------------------------
# Altitude drift compensation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FlightTrack
# ---------------------------------------------------------------------------


class FlightTrack:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Private constructor. Use the factory methods from_igc or from_gopro.
        The dataframe is expected to have standard columns:
        ['timestamp_s', 'x_m', 'y_m', 'z_m', 'weight']
        """
        self._df = dataframe

    @classmethod
    def from_igc(
        cls, igc: IGCTelemetry, origin: WGS84Coordinate = DEFAULT_ORIGIN
    ) -> FlightTrack:
        if igc.data is None:
            raise ValueError("IGC data is None")

        proj_aeqd = pyproj.Proj(
            proj="aeqd", lat_0=origin.lat, lon_0=origin.lon, datum="WGS84", units="m"
        )

        df = igc.data.copy()
        df["x_m"], df["y_m"] = proj_aeqd(
            df["gps_lon_deg"].values, df["gps_lat_deg"].values
        )
        # IGC usually has barometric altitude as primary, but gps_alt is safer for 3D trajectory
        df["z_m"] = df["alt_gps_m"] - origin.alt
        df["weight"] = np.where(df["fix_validity"] == "A", 1.0, 0.01)

        df = sanitize_igc_telemetry(df)
        df = assign_flight_phases(df)
        df = assign_flight_phases_hmm(df)

        return cls(df)

    @classmethod
    def from_gopro(
        cls, gopro: GoProTelemetry, origin: WGS84Coordinate = DEFAULT_ORIGIN
    ) -> FlightTrack:
        if gopro.data is None:
            raise ValueError("GoPro data is None")

        df = gopro.data.copy()

        proj_aeqd = pyproj.Proj(
            proj="aeqd", lat_0=origin.lat, lon_0=origin.lon, datum="WGS84", units="m"
        )

        df["x_m"], df["y_m"] = proj_aeqd(
            df["gps_lon_deg"].values, df["gps_lat_deg"].values
        )
        df["z_m"] = df["gps_alt_m"] - origin.alt
        safe_dop = df["gps_dop"].replace(0.0, 0.1)
        df["weight"] = 1.0 / safe_dop

        df = sanitize_gopro_telemetry(df)
        df = assign_flight_phases(df)
        df = assign_flight_phases_hmm(df)

        return cls(df)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Access the raw, cleaned pandas dataframe."""
        return self._df

    @property
    def t(self) -> npt.NDArray[np.float64]:
        return self._df["timestamp_s"].to_numpy()

    @property
    def points_2d(self) -> npt.NDArray[np.float64]:
        return self._df[["x_m", "y_m"]].to_numpy()

    @property
    def points_3d(self) -> npt.NDArray[np.float64]:
        return self._df[["x_m", "y_m", "z_m"]].to_numpy()

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self._df["weight"].to_numpy()
