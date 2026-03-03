import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flight_composer.config import config
from flight_composer.flight_data import GliderSpecs
from flight_composer.load_flight_data import load_flight_data
from flight_composer.logger import setup_logging


def compute_wing_tips(
    df: pd.DataFrame, half_span: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes left and right wing tip positions in world frame for every row
    in the DataFrame, using the yaw/pitch/roll Euler angles (ZYX convention).

    The wing extends along the body Y-axis: [0, ±half_span, 0].
    Applying R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the second column gives
    the body-Y direction in world coordinates:

        wing_dir_x = cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll)
        wing_dir_y = sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll)
        wing_dir_z = cos(pitch)*sin(roll)

    Returns (left_tips, right_tips) each of shape (N, 3).
    """
    yaw = df["yaw_rad"].values.astype(float)
    pitch = df["pitch_rad"].values.astype(float)
    roll = df["roll_rad"].values.astype(float)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Body-Y axis direction in world frame (second column of R_zyx)
    wing_dx = cy * sp * sr - sy * cr
    wing_dy = sy * sp * sr + cy * cr
    wing_dz = cp * sr

    pos = np.column_stack(
        [
            df["x_m"].values.astype(float),
            df["y_m"].values.astype(float),
            df["z_m"].values.astype(float),
        ]
    )

    wing_dir = np.column_stack([wing_dx, wing_dy, wing_dz])

    right_tips = pos + half_span * wing_dir
    left_tips = pos - half_span * wing_dir

    return left_tips, right_tips


def plot_trajectory_from_parquet(
    df: pd.DataFrame,
    flight_tag: str,
    wingspan_m: float = 15.0,
    wing_bar_interval_s: float = 5.0,
    envelope_sample_s: float = 1.0,
) -> None:
    """
    Creates a 3D visualization of the high-frequency flight trajectory,
    color-coded by flight phase, with a wing envelope overlay showing
    the glider's roll/bank along the path.
    """
    logger = logging.getLogger("flight_actor_data")
    logger.info(f"Visualizing trajectory with {len(df)} points...")

    has_orientation = all(
        col in df.columns for col in ("yaw_rad", "pitch_rad", "roll_rad")
    )
    if not has_orientation:
        logger.warning(
            "Orientation columns (yaw_rad, pitch_rad, roll_rad) not found in "
            "trajectory data. Regenerate actor data to include them. "
            "Skipping wing envelope."
        )

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------------
    # 1. Color-code the trajectory by phase
    # ------------------------------------------------------------------
    unique_phases = df["phase"].unique()
    cmap = plt.get_cmap("tab10")

    for i, phase in enumerate(unique_phases):
        phase_mask = df["phase"] == phase
        phase_indices = np.where(phase_mask)[0]

        if len(phase_indices) == 0:
            continue

        # Detect breaks in contiguous runs (index jumps > 1)
        # and insert NaN separators so ax.plot() breaks the line there.
        breaks = np.where(np.diff(phase_indices) > 1)[0]

        x = df["x_m"].values[phase_indices].astype(float)
        y = df["y_m"].values[phase_indices].astype(float)
        z = df["z_m"].values[phase_indices].astype(float)

        if len(breaks) > 0:
            x = np.insert(x, breaks + 1, np.nan)
            y = np.insert(y, breaks + 1, np.nan)
            z = np.insert(z, breaks + 1, np.nan)

        ax.plot(
            x,
            y,
            z,
            color=cmap(i % 10),
            linewidth=2,
            label=f"{phase}",
        )

    # ------------------------------------------------------------------
    # 2. Draw the wing envelope
    # ------------------------------------------------------------------
    if has_orientation:
        half_span = wingspan_m / 2.0
        timestamps = df["timestamp_s"].values.astype(float)
        dt = np.median(np.diff(timestamps[:1000]))  # Infer sample period

        # --- Envelope ribbon (left & right tip traces) ---
        envelope_step = max(1, int(round(envelope_sample_s / dt)))
        idx_env = np.arange(0, len(df), envelope_step)
        df_env = df.iloc[idx_env]
        left_env, right_env = compute_wing_tips(df_env, half_span)

        ax.plot(
            left_env[:, 0],
            left_env[:, 1],
            left_env[:, 2],
            color="steelblue",
            linewidth=0.5,
            alpha=0.4,
            label="Wing envelope",
        )
        ax.plot(
            right_env[:, 0],
            right_env[:, 1],
            right_env[:, 2],
            color="steelblue",
            linewidth=0.5,
            alpha=0.4,
        )

        # --- Wing cross-bars at regular intervals ---
        bar_step = max(1, int(round(wing_bar_interval_s / dt)))
        idx_bars = np.arange(0, len(df), bar_step)
        df_bars = df.iloc[idx_bars]
        left_bars, right_bars = compute_wing_tips(df_bars, half_span)

        for j in range(len(idx_bars)):
            ax.plot(
                [left_bars[j, 0], right_bars[j, 0]],
                [left_bars[j, 1], right_bars[j, 1]],
                [left_bars[j, 2], right_bars[j, 2]],
                color="steelblue",
                linewidth=0.6,
                alpha=0.5,
            )

        logger.info(
            f"Wing envelope: {len(idx_env)} ribbon samples, "
            f"{len(idx_bars)} cross-bars (wingspan={wingspan_m:.1f}m)."
        )

    # ------------------------------------------------------------------
    # 3. Format the axes
    # ------------------------------------------------------------------
    ax.set_title(f"Smoothed 3D Flight Trajectory: {flight_tag}", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Altitude Z (meters)")

    # Enforce equal aspect ratio so turns and climbs aren't visually distorted
    max_range = (
        np.array(
            [
                df["x_m"].max() - df["x_m"].min(),
                df["y_m"].max() - df["y_m"].min(),
                df["z_m"].max() - df["z_m"].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (df["x_m"].max() + df["x_m"].min()) * 0.5
    mid_y = (df["y_m"].max() + df["y_m"].min()) * 0.5
    mid_z = (df["z_m"].max() + df["z_m"].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Move legend outside the plot area
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def main() -> None:
    setup_logging()
    logger = logging.getLogger("flight_actor_data")

    parser = argparse.ArgumentParser(
        description="Visualize a saved flight trajectory using flight UID",
    )

    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    parser.add_argument(
        "--wingspan_m",
        help="Glider wingspan in meters (default: from GliderSpecs)",
        default=18.0,
        type=float,
    )
    parser.add_argument(
        "--wing_bar_interval",
        help="Seconds between wing cross-bar markers (default: 5.0)",
        default=5.0,
        type=float,
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid
    wingspan_m = (
        args.wingspan_m if args.wingspan_m is not None else GliderSpecs().wingspan_m
    )

    flight = load_flight_data(flight_uid)
    flight_tag = flight.metadata.flight_tag

    trajectory_df_path = config.DIR.ACTOR_DATA / f"{flight_tag}_trajectory_df.parquet"

    if not trajectory_df_path.exists():
        logger.error(
            f"Trajectory parquet file not found: {trajectory_df_path}. "
            "Run generate_actor_data first."
        )
        return

    logger.info(f"Loading trajectory data from {trajectory_df_path}.")
    df = pd.read_parquet(trajectory_df_path, engine="pyarrow")
    logger.info(
        f"Loaded {len(df)} trajectory points for flight [green]{flight_tag}[/green]."
    )

    plot_trajectory_from_parquet(
        df,
        flight_tag,
        wingspan_m=wingspan_m,
        wing_bar_interval_s=args.wing_bar_interval,
    )


if __name__ == "__main__":
    main()
