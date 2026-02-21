#!/usr/bin/env python3
"""
Normalize Telemetry Script

This script processes raw CSV and IGC flight data through Phase I of the trajectory
processing pipeline, generating normalized flight data with synchronized trajectories
and turbulence information.

Usage:
    python normalize_telemetry.py [--flight-number FLIGHT] [--refresh] [--verbose]

Examples:
    python normalize_telemetry.py --flight-number 07
    python normalize_telemetry.py --refresh
    python normalize_telemetry.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from flight_composer import config
from flight_composer.flight_file import find_flight_file, list_available_flights
from flight_composer.trajectory import (
    NormalizedFlightData,
    Phase1Processor,
    load_flight_data,
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def should_process_flight(
    csv_path: Optional[Path],
    igc_path: Optional[Path],
    output_path: Path,
    refresh: bool,
) -> bool:
    """
    Determine if a flight should be processed based on file modification times.

    Args:
        csv_path: Path to the source CSV file (if available)
        igc_path: Path to the source IGC file (if available)
        output_path: Path to the target JSON file
        refresh: Whether to force refresh regardless of modification times

    Returns:
        bool: True if the flight should be processed
    """
    if refresh:
        return True

    if not output_path.exists():
        return True

    # Check if any input file is newer than the output
    try:
        output_mtime = output_path.stat().st_mtime

        if csv_path and csv_path.exists():
            csv_mtime = csv_path.stat().st_mtime
            if csv_mtime > output_mtime:
                return True

        if igc_path and igc_path.exists():
            igc_mtime = igc_path.stat().st_mtime
            if igc_mtime > output_mtime:
                return True

        return False

    except (OSError, AttributeError):
        # If we can't get modification times, process the file
        return True


def generate_output_path(flight_number: str) -> Path:
    """
    Generate the output JSON path for normalized flight data.

    Args:
        flight_number: Flight number

    Returns:
        Path: Expected JSON output path
    """
    json_name = f"{flight_number}_normalized.json"
    return Path(config.TRAJECTORY_DIR) / json_name


def find_input_files(flight_number: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find CSV and IGC input files for a flight.

    Args:
        flight_number: Flight number to search for

    Returns:
        Tuple[Optional[Path], Optional[Path]]: CSV path, IGC path
    """
    # Find CSV file
    csv_path = find_flight_file(flight_number, "csv", [config.TRAJECTORY_CSV_DIR])

    # Find IGC file
    igc_path = find_flight_file(flight_number, "igc", [config.IGC_DIR])

    return csv_path, igc_path


def process_flight_telemetry(
    flight_number: str, refresh: bool, logger: logging.Logger
) -> Tuple[bool, str]:
    """
    Process telemetry data for a single flight through Phase I.

    Args:
        flight_number: Flight number to process
        refresh: Whether to force refresh
        logger: Logger instance

    Returns:
        tuple: (success: bool, status_message: str)
    """
    try:
        # Find input files
        csv_path, igc_path = find_input_files(flight_number)

        if not csv_path and not igc_path:
            return False, f"No input files found for flight {flight_number}"

        if csv_path and not csv_path.exists():
            csv_path = None

        if igc_path and not igc_path.exists():
            igc_path = None

        if not csv_path and not igc_path:
            return False, f"Input files do not exist for flight {flight_number}"

        # Generate output path
        output_path = generate_output_path(flight_number)

        # Check if we should process
        if not should_process_flight(csv_path, igc_path, output_path, refresh):
            return (
                True,
                f"Normalized data for flight {flight_number} is up to date, skipping",
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load input data
        logger.info(f"Processing flight {flight_number}")
        logger.debug(f"CSV source: {csv_path}")
        logger.debug(f"IGC source: {igc_path}")
        logger.debug(f"Output: {output_path}")

        csv_data, igc_data = load_flight_data(flight_number, csv_path, igc_path)

        if csv_data is None and igc_data is None:
            return False, f"Failed to load data for flight {flight_number}"

        # Process through Phase I
        processor = Phase1Processor()
        normalized_data = processor.process_flight(csv_data, igc_data, flight_number)

        if normalized_data is None:
            return False, f"Phase I processing failed for flight {flight_number}"

        # Save normalized data
        normalized_data.save_to_json(output_path)

        # Generate summary statistics
        num_trajectory = len(normalized_data.trajectory_points)
        num_turbulence = len(normalized_data.turbulence_deltas)
        duration = (
            normalized_data.metadata.landing_time
            - normalized_data.metadata.takeoff_time
        )

        return (
            True,
            f"Successfully processed flight {flight_number}: "
            f"{num_trajectory} trajectory points, {num_turbulence} turbulence samples, "
            f"{duration:.1f}s duration",
        )

    except Exception as e:
        logger.error(
            f"Error processing flight {flight_number}: {str(e)}", exc_info=True
        )
        return False, f"Error processing flight {flight_number}: {str(e)}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Normalize telemetry data through Phase I processing"
    )
    parser.add_argument(
        "--flight-number",
        type=str,
        help="Specific flight number to process (processes all if not provided)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration even if JSON file already exists and is newer",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.verbose)

    try:
        # Determine which flights to process
        if args.flight_number:
            flight_numbers = [args.flight_number]
            logger.info(f"Processing flight {args.flight_number}")
        else:
            flight_numbers = list_available_flights()
            if not flight_numbers:
                logger.warning("No flights found in search directories")
                return 0
            logger.info(
                f"Found {len(flight_numbers)} flights: {', '.join(flight_numbers)}"
            )

        # Process flights
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for flight_number in flight_numbers:
            success, message = process_flight_telemetry(
                flight_number, args.refresh, logger
            )

            if success:
                if "skipping" in message:
                    skipped_count += 1
                    logger.debug(message)
                else:
                    processed_count += 1
                    logger.info(message)
            else:
                error_count += 1
                logger.error(message)

        # Summary
        logger.info(
            f"Summary: {processed_count} processed, {skipped_count} skipped, {error_count} errors"
        )

        if error_count > 0 and processed_count == 0:
            logger.error(
                f"Failed to process any flights. {error_count} errors occurred."
            )
            return 1

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
