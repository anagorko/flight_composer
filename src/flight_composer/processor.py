"""
This module implements the data processing pipeline for flight data.
"""

import logging

from flight_composer.config import config
from flight_composer.flight_data import FlightData
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)
from flight_composer.processing.extract_gopro_data import extract_gopro_data
from flight_composer.processing.extract_igc_data import extract_igc_data
from flight_composer.processing.flight_data_sources import find_data_sources


def main():
    setup_logging()
    logger = logging.getLogger("flight_composer.processor")

    logger.info("[yellow]Flight Composer Processor[/yellow]")

    for flight_uid in config.FLIGHT_UIDS:
        print()

        ### 1. Source files discovery

        flight_data_sources = find_data_sources(flight_uid)
        flight_tag = flight_data_sources.flight_tag

        ### 2. MP4 telemetry data extraction
        if flight_data_sources.gopro_path:
            logger.info(
                f"Extracting [bold]GoPro[/bold] telemetry data from {flight_data_sources.gopro_path}"
            )

            gopro_telemetry = extract_gopro_data(flight_data_sources.gopro_path)
            if gopro_telemetry:
                logger.info(
                    f"Extracted MP4 telemetry data: [green]{get_gopro_overlay_text(gopro_telemetry)}[/green]"
                )
        else:
            gopro_telemetry = None

        if flight_data_sources.igc_path:
            logger.info(
                f"Extracting [bold]IGC[/bold] telemetry data from {flight_data_sources.igc_path}"
            )

            igc_telemetry = extract_igc_data(flight_data_sources.igc_path)
            if igc_telemetry:
                logger.info(
                    f"Extracted IGC telemetry data: [green]{get_flight_map_overlay_text(igc_telemetry)}[/green]"
                )
        else:
            igc_telemetry = None

        _ = FlightData(
            flight_uid=flight_uid,
            flight_tag=flight_tag,
            gopro=gopro_telemetry,
            gopro_path=flight_data_sources.gopro_path,
            igc=igc_telemetry,
            igc_path=flight_data_sources.igc_path,
        )


if __name__ == "__main__":
    main()
