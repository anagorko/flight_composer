import logging
from enum import StrEnum

import numpy as np
import pandas as pd

from flight_composer.flight_data import Airfield

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


def assign_flight_phases(validated_df: pd.DataFrame) -> pd.DataFrame:
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


def assign_flight_phases_hmm(
    validated_df: pd.DataFrame, airfield: Airfield
) -> pd.DataFrame:
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

    # z_agl: altitude above ground level derived from known airfield elevation
    ground_z = float(airfield.origin.alt)
    # Clip negative values to 0.0 to prevent massive HMM penalties on the ground
    z_agl = np.maximum(z - ground_z, 0.0)

    logger.info(
        f"Ground level z={ground_z:.1f} m (provided by airfield {airfield.icao_code})"
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
            "sigma": np.array([  5.0,   2.0,   30.0,   5.0])},   # PRE_FLIGHT
        1: {"mu": np.array([ 10.0,   0.5,    1.5,   1.0]),
            "sigma": np.array([  6.0,   1.5,   30.0,   3.0])},   # LAUNCH_ROLL
        2: {"mu": np.array([ 28.0,   5.0,  800.0,   2.0]),
            "sigma": np.array([ 10.0,   3.0, 1000.0,   4.0])},   # LAUNCH_CLIMB
        3: {"mu": np.array([ 25.0,  -0.5,  800.0,   2.0]),
            "sigma": np.array([ 10.0,   2.0, 1000.0,   4.0])},   # CRUISING
        4: {"mu": np.array([ 22.0,   0.5,  800.0,  18.0]),
            "sigma": np.array([ 10.0,   3.0, 1000.0,   8.0])},   # THERMALLING
        5: {"mu": np.array([  8.0,  -0.5,    3.0,   2.0]),
            "sigma": np.array([  8.0,   2.0,   30.0,   4.0])},   # LANDING_ROLL
        6: {"mu": np.array([  0.5,   0.0,    0.0,   0.0]),
            "sigma": np.array([  5.0,   2.0,   30.0,   5.0])},   # POST_FLIGHT
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

    # Initialize by allowing the flight to start in ANY phase.
    # We apply a uniform prior probability (1/NUM_STATES) to all starting states.
    V[0, :] = np.log(1.0 / NUM_STATES) + log_B[0, :]

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
