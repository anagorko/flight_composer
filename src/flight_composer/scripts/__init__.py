"""
Flight Composer Scripts Package

This package contains command-line scripts for converting flight data files
to various formats used by the flight trajectory interpolation algorithm.

Available scripts:
- gpx_to_geojson: Convert GPX files to GeoJSON format
- igc_to_geojson: Convert IGC files to GeoJSON format
"""

__version__ = "1.0.0"
__all__ = ["gpx_to_geojson", "igc_to_geojson"]
