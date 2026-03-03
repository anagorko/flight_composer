import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flight_composer.flight_track import FlightTrack
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)
from flight_composer.processing.extract_gopro_data import extract_gopro_data
from flight_composer.processing.extract_igc_data import extract_igc_data
from flight_composer.processing.flight_data_sources import find_data_sources

# Map the string values of your FlightPhase enum to distinct colors
PHASE_COLORS = {
    "PRE_FLIGHT": "gray",
    "LAUNCH_ROLL": "orange",
    "LAUNCH_CLIMB": "red",
    "CRUISING": "blue",
    "THERMALLING": "green",
    "ROLLING_LANDING": "purple",  # Note: Enum value for LANDING_ROLL
    "PUSHING_GROUND": "brown",  # Note: Enum value for POST_FLIGHT
}


def plot_track_altitude(ax: plt.Axes, dataframe: pd.DataFrame, title: str) -> None:
    """Plots altitude over time on the given axes, colored by flight phase."""
    t = dataframe["timestamp_s"].to_numpy()
    z = dataframe["z_m"].to_numpy()
    phases = dataframe["phase_hmm"].to_numpy()

    # Draw a faint connecting line underneath the scatter points
    ax.plot(t, z, color="lightgray", linewidth=1, zorder=1)

    # Plot each phase as uniquely colored scatter points for a clean legend
    for phase_val, color in PHASE_COLORS.items():
        mask = phases == phase_val
        if not mask.any():
            continue

        ax.scatter(
            t[mask],
            z[mask],
            color=color,
            label=phase_val,
            s=4,  # Point size
            zorder=2,  # Keep points above the connecting line
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Altitude Relative to Origin (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Place legend outside the plot area so it doesn't obscure the flight track
    ax.legend(title="Flight Phase", bbox_to_anchor=(1.02, 1), loc="upper left")


def main() -> None:
    setup_logging()
    logger = logging.getLogger("flight_altitude_plot")

    parser = argparse.ArgumentParser(
        description="Plot altitude over time with flight phases using flight UID",
    )
    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid
    flight_data_sources = find_data_sources(flight_uid)
    flight_tag = flight_data_sources.flight_tag

    logger.info(f"Processing flight [green]{flight_tag}[/green].")

    tracks_to_plot = []

    # 1. Load GoPro data
    if flight_data_sources.gopro_path:
        logger.info(f"Found GoPro data: {flight_data_sources.gopro_path}")
        gopro_telemetry = extract_gopro_data(flight_data_sources.gopro_path)
        if gopro_telemetry:
            logger.info(
                f"Extracted GoPro telemetry: [green]{get_gopro_overlay_text(gopro_telemetry)}[/green]"
            )
            gopro_track = FlightTrack.from_gopro(gopro_telemetry)
            tracks_to_plot.append(("GoPro Track", gopro_track))
    else:
        logger.info("No GoPro data found.")

    # 2. Load IGC data
    if flight_data_sources.igc_path:
        logger.info(f"Found IGC data: {flight_data_sources.igc_path}")
        igc_telemetry = extract_igc_data(flight_data_sources.igc_path)
        if igc_telemetry:
            logger.info(
                f"Extracted IGC telemetry: [green]{get_flight_map_overlay_text(igc_telemetry)}[/green]"
            )
            igc_track = FlightTrack.from_igc(igc_telemetry)
            tracks_to_plot.append(("IGC Track", igc_track))
    else:
        logger.info("No IGC data found.")

    if not tracks_to_plot:
        logger.warning(
            f"No valid telemetry data found for flight {flight_tag} to plot."
        )
        return

    # 3. Create Figure and Subplots
    num_plots = len(tracks_to_plot)
    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=1,
        figsize=(12, 5 * num_plots),
        squeeze=False,
        constrained_layout=True,  # Handles legend bounding boxes neatly
    )

    for idx, (source_name, track) in enumerate(tracks_to_plot):
        ax = axes[idx, 0]
        title = f"{flight_tag} - {source_name}"
        plot_track_altitude(ax, track.dataframe, title)

    # 4. Save to SVG
    output_filename = f"{flight_tag}_alt.svg"
    plt.savefig(output_filename, format="svg", bbox_inches="tight")
    logger.info(f"Successfully saved altitude plot to [green]{output_filename}[/green]")


if __name__ == "__main__":
    main()
