import argparse
import logging

from flight_composer.config import config
from flight_composer.flight_track import FlightTrack
from flight_composer.load_flight_data import load_flight_data
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)
from flight_composer.phase_subtitles import generate_phase_srt


def main() -> None:
    setup_logging()
    logger = logging.getLogger("flight_phases")

    parser = argparse.ArgumentParser(
        description="Create phase subtitles for flight recorded with GoPro using flight UID",
    )

    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid

    flight = load_flight_data(flight_uid)
    flight_tag = flight.metadata.flight_tag

    logger.info(f"Processing flight [green]{flight_tag}[/green].")

    # GoPro data
    if flight.gopro:
        logger.info(f"Found GoPro data, {flight.gopro_path}.")
        logger.info(
            f"Extracted MP4 telemetry data: [green]{get_gopro_overlay_text(flight.gopro)}[/green]. Fitting GoPro spline."
        )

        gopro_track = FlightTrack.from_gopro(flight)
        config.DIR.SUBTITLES.mkdir(parents=True, exist_ok=True)
        gopro_srt_path = config.DIR.SUBTITLES / f"{flight_tag}_gopro_phases.srt"
        generate_phase_srt(
            gopro_track.dataframe,
            str(gopro_srt_path),
            time_col="timestamp_s",
            phase_col="phase_hmm",
        )
        logger.info(f"Saved GoPro phase subtitles to {gopro_srt_path}")
    else:
        logger.info("No GoPro data found.")

    # IGC data
    if flight.igc:
        logger.info(f"Found IGC data, {flight.igc_path}.")
        logger.info(
            f"Extracted IGC telemetry data: [green]{get_flight_map_overlay_text(flight.igc)}[/green]"
        )

        # TODO: makes no sense without time_delta to gopro track

        igc_track = FlightTrack.from_igc(flight)
        config.DIR.SUBTITLES.mkdir(parents=True, exist_ok=True)
        igc_srt_path = config.DIR.SUBTITLES / f"{flight_tag}_igc_phases.srt"
        generate_phase_srt(
            igc_track.dataframe,
            str(igc_srt_path),
            time_col="timestamp_s",
            phase_col="phase_hmm",
        )
        logger.info(f"Saved IGC phase subtitles to {igc_srt_path}")
    else:
        logger.info("No IGC data found.")


if __name__ == "__main__":
    main()
