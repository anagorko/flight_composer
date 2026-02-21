#!/usr/bin/env python3
"""
IGC to GeoJSON Converter Script

This script reads an IGC file containing flight track data and converts it to a GeoJSON
LineString with elevation annotations. The output follows the GeoJSON specification
and is saved to the PROCESSED_DATA_DIR directory.

Usage:
    python igc_to_geojson.py <input_igc_file>

Example:
    python igc_to_geojson.py 07_niskie_ladowanie.igc
"""

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from flight_composer import flight_file
from flight_composer.trajectory import config


def parse_igc_coordinate(coord_str, is_longitude=False):
    """
    Parse IGC coordinate format to decimal degrees.

    IGC format examples:
    - Latitude: 5216203N (52°16.203'N)
    - Longitude: 02054885E (020°54.885'E)

    Args:
        coord_str (str): IGC coordinate string
        is_longitude (bool): True if parsing longitude, False for latitude

    Returns:
        float: Decimal degrees
    """
    if is_longitude:
        # Longitude format: DDDMMMMM[EW]
        if len(coord_str) < 9:
            raise ValueError(f"Invalid longitude format: {coord_str}")
        degrees = int(coord_str[:3])
        minutes = int(coord_str[3:8]) / 1000.0
        direction = coord_str[8]
    else:
        # Latitude format: DDMMMMM[NS]
        if len(coord_str) < 8:
            raise ValueError(f"Invalid latitude format: {coord_str}")
        degrees = int(coord_str[:2])
        minutes = int(coord_str[2:7]) / 1000.0
        direction = coord_str[7]

    decimal = degrees + minutes / 60.0

    if direction in ["S", "W"]:
        decimal = -decimal

    return decimal


def parse_igc_time(time_str):
    """
    Parse IGC time format (HHMMSS) to seconds since midnight.

    Args:
        time_str (str): IGC time string (HHMMSS)

    Returns:
        int: Seconds since midnight
    """
    if len(time_str) != 6:
        raise ValueError(f"Invalid time format: {time_str}")

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])

    return hours * 3600 + minutes * 60 + seconds


def parse_igc_file(igc_file_path):
    """
    Parse an IGC file and extract track points.

    Args:
        igc_file_path (Path): Path to the IGC file

    Returns:
        dict: Dictionary containing track points and metadata
    """
    track_points = []
    metadata = {
        "pilot": None,
        "glider_type": None,
        "glider_id": None,
        "date": None,
        "competition_id": None,
        "competition_class": None,
    }

    flight_date = None

    with open(igc_file_path, "r", encoding="latin-1") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse header records
                if line.startswith("HFDTE"):
                    # Date record: HFDTEDATE:DDMMYY,XX
                    match = re.search(r"DATE:(\d{6})", line)
                    if match:
                        date_str = match.group(1)
                        day = int(date_str[:2])
                        month = int(date_str[2:4])
                        year = 2000 + int(date_str[4:6])  # Assume 20xx
                        flight_date = date(year, month, day)
                        metadata["date"] = flight_date.isoformat()

                elif line.startswith("HFPLT"):
                    # Pilot record
                    match = re.search(r"PILOTINCHARGE:(.+)", line)
                    if match:
                        metadata["pilot"] = match.group(1).strip()

                elif line.startswith("HFGTY"):
                    # Glider type record
                    match = re.search(r"GLIDERTYPE:(.+)", line)
                    if match:
                        metadata["glider_type"] = match.group(1).strip()

                elif line.startswith("HFGID"):
                    # Glider ID record
                    match = re.search(r"GLIDERID:(.+)", line)
                    if match:
                        metadata["glider_id"] = match.group(1).strip()

                elif line.startswith("HFCID"):
                    # Competition ID record
                    match = re.search(r"COMPETITIONID:(.+)", line)
                    if match:
                        metadata["competition_id"] = match.group(1).strip()

                elif line.startswith("HFCCL"):
                    # Competition class record
                    match = re.search(r"COMPETITIONCLASS:(.+)", line)
                    if match:
                        metadata["competition_class"] = match.group(1).strip()

                # Parse B records (fix records)
                elif line.startswith("B"):
                    if len(line) < 35:
                        continue  # Skip malformed B records

                    # B record format: BHHMMSSDDMMMMMNSDDMMMMMEWAAAAAAAAAA
                    time_str = line[1:7]  # HHMMSS
                    lat_str = line[7:15]  # DDMMMMM[NS]
                    lon_str = line[15:24]  # DDDMMMMM[EW]
                    validity = line[24]  # A=3D valid, V=invalid

                    # Skip invalid fixes
                    if validity != "A":
                        continue

                    # Parse coordinates
                    latitude = parse_igc_coordinate(lat_str, is_longitude=False)
                    longitude = parse_igc_coordinate(lon_str, is_longitude=True)

                    # Parse altitudes (if available)
                    if len(line) >= 35:
                        baro_alt_str = line[25:30]
                        gps_alt_str = line[30:35]

                        try:
                            baro_altitude = (
                                int(baro_alt_str)
                                if baro_alt_str.isdigit()
                                or (
                                    baro_alt_str[0] == "-"
                                    and baro_alt_str[1:].isdigit()
                                )
                                else None
                            )
                            gps_altitude = (
                                int(gps_alt_str)
                                if gps_alt_str.isdigit()
                                or (gps_alt_str[0] == "-" and gps_alt_str[1:].isdigit())
                                else None
                            )
                        except (ValueError, IndexError):
                            baro_altitude = None
                            gps_altitude = None
                    else:
                        baro_altitude = None
                        gps_altitude = None

                    # Use GPS altitude if available, otherwise barometric
                    elevation = (
                        gps_altitude if gps_altitude is not None else baro_altitude
                    )
                    if elevation is None:
                        elevation = 0.0

                    # Create timestamp
                    time_seconds = parse_igc_time(time_str)
                    timestamp = None
                    if flight_date:
                        timestamp = datetime.combine(
                            flight_date,
                            datetime.min.time().replace(
                                hour=time_seconds // 3600,
                                minute=(time_seconds % 3600) // 60,
                                second=time_seconds % 60,
                            ),
                            tzinfo=timezone.utc,
                        ).isoformat()

                    track_point = {
                        "longitude": longitude,
                        "latitude": latitude,
                        "elevation": float(elevation),
                        "time": timestamp,
                        "baro_altitude": baro_altitude,
                        "gps_altitude": gps_altitude,
                    }
                    track_points.append(track_point)

            except Exception as e:
                print(f"Warning: Error parsing line {line_num}: {line[:50]}... - {e}")
                continue

    return {"track_points": track_points, "metadata": metadata}


def create_geojson_linestring(track_points, metadata):
    """
    Create a GeoJSON LineString from track points with elevation annotations.

    Args:
        track_points (list): List of track points with coordinates and metadata
        metadata (dict): IGC file metadata

    Returns:
        dict: GeoJSON feature collection
    """
    if not track_points:
        raise ValueError("No track points found in IGC file")

    # Extract coordinates for LineString (lon, lat, elevation)
    coordinates = []
    for point in track_points:
        coord = [point["longitude"], point["latitude"], point["elevation"]]
        coordinates.append(coord)

    # Calculate additional properties
    total_points = len(track_points)
    start_time = track_points[0]["time"] if track_points[0]["time"] else None
    end_time = track_points[-1]["time"] if track_points[-1]["time"] else None

    # Calculate elevation statistics
    elevations = [point["elevation"] for point in track_points]
    min_elevation = min(elevations)
    max_elevation = max(elevations)
    avg_elevation = sum(elevations) / len(elevations)

    # Create properties from metadata
    properties = {
        "name": "Flight Track",
        "description": f"Flight track with {total_points} points",
        "total_points": total_points,
        "start_time": start_time,
        "end_time": end_time,
        "elevation_stats": {
            "min_elevation": round(min_elevation, 2),
            "max_elevation": round(max_elevation, 2),
            "avg_elevation": round(avg_elevation, 2),
        },
        "source": "IGC",
        "coordinate_system": "WGS84",
    }

    # Add metadata to properties if available
    if metadata["pilot"]:
        properties["pilot"] = metadata["pilot"]
    if metadata["glider_type"]:
        properties["glider_type"] = metadata["glider_type"]
    if metadata["glider_id"]:
        properties["glider_id"] = metadata["glider_id"]
    if metadata["date"]:
        properties["flight_date"] = metadata["date"]
    if metadata["competition_id"]:
        properties["competition_id"] = metadata["competition_id"]
    if metadata["competition_class"]:
        properties["competition_class"] = metadata["competition_class"]

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": properties,
            }
        ],
    }

    return geojson


def save_geojson(geojson_data, output_path):
    """
    Save GeoJSON data to file.

    Args:
        geojson_data (dict): GeoJSON data structure
        output_path (Path): Output file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"GeoJSON file saved to: {output_path}")


def main():
    """Main function to handle command line arguments and orchestrate conversion."""
    parser = argparse.ArgumentParser(
        description="Convert IGC file to GeoJSON format using flight number",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  igc-to-geojson 7                    # Convert flight #7 IGC file
  igc-to-geojson 142                  # Convert flight #142 IGC file
  igc-to-geojson /path/to/flight.igc  # Convert specific file
        """,
    )

    parser.add_argument(
        "input",
        help="Flight number (e.g., 7, 142) or full path to IGC file",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output GeoJSON file path (default: auto-generated in PROCESSED_DATA_DIR)",
    )

    args = parser.parse_args()

    # Determine input file path
    input_path = None

    # Check if input is a path (contains / or \) or an existing file
    if "/" in args.input or "\\" in args.input or Path(args.input).exists():
        # Treat as file path
        input_path = Path(args.input)
        if not input_path.is_absolute() and not input_path.exists():
            # Try to find the file in TRAJECTORY_DIR
            input_path = config.TRAJECTORY_DIR / args.input
    else:
        # Treat as flight number
        try:
            flight_number = args.input
            input_path = flight_file.find_flight_file(flight_number, "igc")
            if input_path is None:
                print(f"Error: No IGC file found for flight #{flight_number}")
                print("Available flights:")
                available_flights = flight_file.list_available_flights()
                for flight in available_flights:
                    print(f"  {flight}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid flight number: {args.input}")
            sys.exit(1)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename with _igc suffix
        stem = input_path.stem
        output_path = config.PROCESSED_DATA_DIR / f"{stem}_igc.geojson"

    try:
        print(f"Reading IGC file: {input_path}")
        parsed_data = parse_igc_file(input_path)
        track_points = parsed_data["track_points"]
        metadata = parsed_data["metadata"]

        if not track_points:
            print("Error: No valid track points found in IGC file")
            sys.exit(1)

        print(f"Found {len(track_points)} track points")

        print("Converting to GeoJSON...")
        geojson_data = create_geojson_linestring(track_points, metadata)

        save_geojson(geojson_data, output_path)

        # Print summary
        feature = geojson_data["features"][0]
        props = feature["properties"]
        print("\nConversion Summary:")
        print(f"  Total points: {props['total_points']}")
        print(
            f"  Elevation range: {props['elevation_stats']['min_elevation']:.2f}m - {props['elevation_stats']['max_elevation']:.2f}m"
        )
        print(f"  Average elevation: {props['elevation_stats']['avg_elevation']:.2f}m")
        if props.get("pilot"):
            print(f"  Pilot: {props['pilot']}")
        if props.get("glider_type"):
            print(f"  Glider: {props['glider_type']}")
        if props.get("flight_date"):
            print(f"  Date: {props['flight_date']}")
        if props.get("start_time") and props.get("end_time"):
            print(f"  Time range: {props['start_time']} to {props['end_time']}")

    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
