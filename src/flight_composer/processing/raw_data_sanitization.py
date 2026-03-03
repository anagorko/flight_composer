import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        dropped_timestamps = df.loc[~monotonic_mask, "timestamp_s"].values
        logger.info(
            f"IGC sanitization: dropped {dropped}/{original_count} rows "
            f"with non-strictly-increasing timestamps."
        )
        logger.info(
            f"IGC sanitization: dropped point timestamps: "
            f"{', '.join(f'{ts:.2f}s' for ts in dropped_timestamps)}"
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

                    if "weight" in df.columns:
                        df.loc[bad_indices, "weight"] = 0.01

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

                    if "weight" in df.columns:
                        df.loc[climb_mask, "weight"] = 0.01

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

    mono_dropped = original_count - monotonic_mask.sum()
    if mono_dropped > 0:
        dropped_timestamps = df.loc[~monotonic_mask, "timestamp_s"].values
        logger.info(
            f"GoPro sanitization: dropped {mono_dropped}/{original_count} rows "
            f"with non-strictly-increasing timestamps."
        )
        logger.info(
            f"GoPro sanitization: dropped point timestamps: "
            f"{', '.join(f'{ts:.2f}s' for ts in dropped_timestamps)}"
        )

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

        kinematic_dropped_timestamps = df_clean.loc[
            dropped_indices, "timestamp_s"
        ].values
        logger.info(
            f"GoPro sanitization: dropped point timestamps (kinematics): "
            f"{', '.join(f'{ts:.2f}s' for ts in kinematic_dropped_timestamps)}"
        )

    return df_clean[valid_mask].reset_index(drop=True)
