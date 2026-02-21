#!/usr/bin/env python3
"""
Export trajectory data with WGS84 coordinates for Cesium integration.

This script exports flight trajectory data preserving the original WGS84 coordinates
(latitude, longitude, altitude) instead of converting to local ENU coordinates.
This allows Unreal Engine with Cesium to perform the coordinate transformation.

Usage:
    python export_trajectory_wgs84.py <flight_number>

Example:
    python export_trajectory_wgs84.py 07
"""

import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def export_trajectory_wgs84(flight_number):
    """Export trajectory with WGS84 coordinates for Cesium integration."""
    try:
        from flight_composer.flight_file import find_flight_file
        from flight_composer.trajectory import process_flight_phase_i
        from flight_composer.trajectory.loaders import (
            GeoJSONLoader,
            GPXLoader,
            IGCLoader,
        )

        print(f"Processing flight {flight_number} to extract WGS84 coordinates...")

        # First, process through Phase I to get normalized data
        result = process_flight_phase_i(flight_number)

        print(f"Phase I completed: {len(result.trajectory_points)} points processed")

        # Now load the original raw data to get WGS84 coordinates
        igc_path = find_flight_file(flight_number, "igc")
        gpx_path = find_flight_file(flight_number, "gpx")
        geojson_path = find_flight_file(flight_number, "geojson")

        # Load original coordinates from available sources
        original_points = []

        if geojson_path and geojson_path.exists():
            print(f"Loading original coordinates from GeoJSON: {geojson_path}")
            loader = GeoJSONLoader()
            geojson_points = loader.load(geojson_path)
            original_points.extend(geojson_points)

        if igc_path and igc_path.exists() and not original_points:
            print(f"Loading original coordinates from IGC: {igc_path}")
            loader = IGCLoader()
            igc_points = loader.load(igc_path)
            original_points.extend(igc_points)

        if gpx_path and gpx_path.exists() and not original_points:
            print(f"Loading original coordinates from GPX: {gpx_path}")
            loader = GPXLoader()
            gpx_points = loader.load(gpx_path)
            original_points.extend(gpx_points)

        if not original_points:
            raise ValueError(
                f"No original coordinate data found for flight {flight_number}"
            )

        print(f"Loaded {len(original_points)} original coordinate points")

        # Convert to JSON-serializable format (keeping WGS84 coordinates)
        trajectory_data = []
        for point in original_points:
            # Safely extract attributes with proper None handling
            quality = getattr(point, "quality", 1.0)
            if quality is None:
                quality = 1.0

            ground_speed = getattr(point, "ground_speed_kmh", 0.0)
            if ground_speed is None:
                ground_speed = 0.0

            hdop = getattr(point, "hdop", 0.0)
            if hdop is None:
                hdop = 0.0

            trajectory_data.append(
                {
                    "timestamp": float(point.timestamp),
                    "latitude": float(point.latitude),
                    "longitude": float(point.longitude),
                    "altitude_msl": float(point.altitude_msl),
                    "source": point.source,
                    "quality": float(quality),
                    "ground_speed_kmh": float(ground_speed),
                    "hdop": float(hdop),
                }
            )

        # Calculate duration
        duration = original_points[-1].timestamp - original_points[0].timestamp

        # Create output data
        output_data = {
            "flight_number": flight_number,
            "coordinate_system": "WGS84",
            "duration": float(duration),
            "total_points": len(trajectory_data),
            "origin": {
                "latitude": float(result.metadata.origin.latitude),
                "longitude": float(result.metadata.origin.longitude),
                "altitude_msl": float(result.metadata.origin.altitude_msl),
            },
            "bounds": {
                "min_latitude": float(min(p["latitude"] for p in trajectory_data)),
                "max_latitude": float(max(p["latitude"] for p in trajectory_data)),
                "min_longitude": float(min(p["longitude"] for p in trajectory_data)),
                "max_longitude": float(max(p["longitude"] for p in trajectory_data)),
                "min_altitude": float(min(p["altitude_msl"] for p in trajectory_data)),
                "max_altitude": float(max(p["altitude_msl"] for p in trajectory_data)),
            },
            "sources": list(set(p["source"] for p in trajectory_data)),
            "trajectory_points": trajectory_data,
        }

        # Save to JSON file
        output_path = Path("ProcessedData") / f"flight_{flight_number}_wgs84.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Exported {len(trajectory_data):,} points to {output_path}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Coordinate system: WGS84 (for Cesium integration)")
        print(f"   Sources: {', '.join(output_data['sources'])}")

        # Display bounds
        bounds = output_data["bounds"]
        print(
            f"   Latitude: {bounds['min_latitude']:.6f}° to {bounds['max_latitude']:.6f}°"
        )
        print(
            f"   Longitude: {bounds['min_longitude']:.6f}° to {bounds['max_longitude']:.6f}°"
        )
        print(
            f"   Altitude: {bounds['min_altitude']:.1f}m to {bounds['max_altitude']:.1f}m MSL"
        )

        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python export_trajectory_wgs84.py <flight_number>")
        print("Example: python export_trajectory_wgs84.py 07")
        sys.exit(1)

    flight_number = sys.argv[1]
    print(f"WGS84 Trajectory Export for Flight {flight_number}")
    print("=" * 50)

    success = export_trajectory_wgs84(flight_number)

    if success:
        print(f"\n🎉 Export completed successfully!")
        print(
            f"   The WGS84 coordinate file is ready for Cesium integration in Unreal Engine."
        )
        print(f"   Next steps:")
        print(f"   1. Open EPBC project in Unreal Engine")
        print(f"   2. Add TrajectoryTestActor to your level")
        print(f"   3. Enable 'Use Cesium Coordinates'")
        print(f"   4. Set Flight Number to '{flight_number}'")
        print(f"   5. Click 'Load Trajectory Data'")
    else:
        print(f"\n❌ Export failed - check error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()
