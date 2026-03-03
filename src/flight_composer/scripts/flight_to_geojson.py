import argparse
import json
import logging

import numpy as np
import pyproj
from shapely.geometry import LineString, mapping
from shapely.ops import transform

from flight_composer.config import config
from flight_composer.flight_track import FlightTrack
from flight_composer.kinematic_spline import KinematicSpline
from flight_composer.load_flight_data import load_flight_data
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)
from flight_composer.projection import DEFAULT_ORIGIN


def export_flight_geojson(tracks_to_export: list[dict], filename: str):
    """
    Exports multiple Shapely geometries to a single styled GeoJSON file.

    Expected format for tracks_to_export:
    [
        {"geometry": LineString(...), "name": "GoPro Smooth", "color": "#0000ff"},
        ...
    ]
    """
    features = []

    for track in tracks_to_export:
        geom = track.get("geometry")
        if not geom or geom.is_empty:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": track.get("name", "Flight Track"),
                    "stroke": track.get("color", "#555555"),  # simplestyle-spec color
                    "stroke-width": track.get("width", 3),  # simplestyle-spec width
                    "stroke-opacity": track.get("opacity", 1.0),
                },
                "geometry": mapping(geom),
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    with open(filename, "w") as f:
        json.dump(geojson, f, indent=2)


def project_to_wgs84(local_linestring: LineString, origin) -> LineString:
    """Helper to convert local Cartesian LineString back to WGS84 (Lon, Lat)."""
    local_crs = pyproj.CRS(
        f"+proj=aeqd +lat_0={origin.lat} +lon_0={origin.lon} +datum=WGS84 +units=m"
    )
    wgs84_crs = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(local_crs, wgs84_crs, always_xy=True)
    return transform(transformer.transform, local_linestring)


def main() -> None:
    setup_logging()
    logger = logging.getLogger("flight_to_geojson")

    parser = argparse.ArgumentParser(
        description="Convert flight GoPro/IGC data to GeoJSON format using flight UID",
    )

    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid

    flight = load_flight_data(flight_uid)
    flight_tag = flight.metadata.flight_tag

    logger.info(f"Processing flight [green]{flight_tag}[/green].")

    geojson_layers: list[dict] = []

    # GoPro data
    if flight.gopro:
        logger.info(f"Found GoPro data, {flight.gopro_path}.")
        logger.info(
            f"Extracted MP4 telemetry data: [green]{get_gopro_overlay_text(flight.gopro)}[/green]. Fitting GoPro spline."
        )

        gopro_track = FlightTrack.from_gopro(flight)

        gopro_spline = KinematicSpline(degree=5, smoothing=len(gopro_track.t) * 0.1)
        gopro_spline.fit(gopro_track.t, gopro_track.points_2d, gopro_track.weights)

        # Evaluate and project
        t_gopro = np.arange(gopro_spline.t_begin, gopro_spline.t_end, 0.02)
        gopro_smooth_xy = gopro_spline(t_gopro, derivative=0)

        gopro_wgs84 = project_to_wgs84(LineString(gopro_smooth_xy), DEFAULT_ORIGIN)
        raw_gopro_wgs84 = project_to_wgs84(
            LineString(gopro_track.points_2d), DEFAULT_ORIGIN
        )

        geojson_layers.append(
            {
                "geometry": raw_gopro_wgs84,
                "name": "Raw GoPro",
                "color": "#ff0000",  # Red for raw
                "width": 1,
                "opacity": 0.5,
            }
        )
        geojson_layers.append(
            {
                "geometry": gopro_wgs84,
                "name": "Smooth GoPro",
                "color": "#0000ff",  # Blue for smooth GoPro
            }
        )
    else:
        logger.info("No GoPro data found.")

    # IGC data
    if flight.igc:
        logger.info(f"Found IGC data, {flight.igc_path}.")
        logger.info(
            f"Extracted IGC telemetry data: [green]{get_flight_map_overlay_text(flight.igc)}[/green]"
        )

        igc_track = FlightTrack.from_igc(flight)
        igc_spline = KinematicSpline(degree=5, smoothing=len(igc_track.t) * 0.01)
        igc_spline.fit(igc_track.t, igc_track.points_2d, igc_track.weights)

        # Evaluate and project
        t_igc = np.arange(igc_spline.t_begin, igc_spline.t_end, 0.02)
        igc_smooth_xy = igc_spline(t_igc, derivative=0)

        igc_wgs84 = project_to_wgs84(LineString(igc_smooth_xy), DEFAULT_ORIGIN)
        raw_igc_wgs84 = project_to_wgs84(
            LineString(igc_track.points_2d), DEFAULT_ORIGIN
        )

        geojson_layers.append(
            {
                "geometry": raw_igc_wgs84,
                "name": "Raw IGC",
                "color": "#ffaa00",  # Orange for raw IGC
                "width": 1,
                "opacity": 0.5,
            }
        )
        geojson_layers.append(
            {
                "geometry": igc_wgs84,
                "name": "Smooth IGC",
                "color": "#00ff00",  # Green for smooth IGC
            }
        )
    else:
        logger.info("No IGC data found.")

    config.DIR.GEOJSON.mkdir(parents=True, exist_ok=True)
    output_path = config.DIR.GEOJSON / f"{flight_tag}.geojson"
    logger.info(f"Exporting to {output_path}...")
    export_flight_geojson(geojson_layers, str(output_path))


if __name__ == "__main__":
    main()
