#!/usr/bin/env python3
"""
Test script for trajectory data export functionality.

This script tests the Phase I processing and JSON export that will be used
by the Unreal Engine TrajectoryTestActor.

Usage:
    python test_trajectory_export.py [flight_number]

Example:
    python test_trajectory_export.py 07
"""

import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_trajectory_export(flight_number):
    """Test trajectory export for a specific flight."""
    try:
        from flight_composer.trajectory import process_flight_phase_i

        print(f"Testing trajectory export for flight {flight_number}...")

        # Process flight data
        print("Running Phase I processing...")
        result = process_flight_phase_i(flight_number)

        print(f"✅ Phase I completed successfully!")
        print(f"   Duration: {result.duration:.1f} seconds")
        print(f"   Points: {len(result.trajectory_points):,}")
        print(
            f"   Origin: {result.metadata.origin.latitude:.6f}°, {result.metadata.origin.longitude:.6f}°"
        )

        # Convert to JSON-serializable format
        print("Converting to JSON format...")
        trajectory_data = []
        for point in result.trajectory_points:
            trajectory_data.append(
                {
                    "timestamp": float(point.timestamp),
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                    "source": point.source,
                    "quality": float(point.quality),
                }
            )

        # Create output data
        output_data = {
            "flight_number": result.metadata.flight_number or flight_number,
            "duration": float(result.duration),
            "total_points": len(result.trajectory_points),
            "origin": {
                "latitude": float(result.metadata.origin.latitude),
                "longitude": float(result.metadata.origin.longitude),
                "altitude_msl": float(result.metadata.origin.altitude_msl),
            },
            "trajectory_points": trajectory_data,
        }

        # Save to JSON file
        output_path = Path("ProcessedData") / f"flight_{flight_number}_trajectory.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Exported {len(trajectory_data):,} points to {output_path}")

        # Display some statistics
        if trajectory_data:
            timestamps = [p["timestamp"] for p in trajectory_data]
            x_coords = [p["x"] for p in trajectory_data]
            y_coords = [p["y"] for p in trajectory_data]
            z_coords = [p["z"] for p in trajectory_data]

            print(f"\n📊 Trajectory Statistics:")
            print(f"   Time range: {min(timestamps):.1f} - {max(timestamps):.1f} s")
            print(f"   X (East) range: {min(x_coords):.1f} - {max(x_coords):.1f} m")
            print(f"   Y (North) range: {min(y_coords):.1f} - {max(y_coords):.1f} m")
            print(f"   Z (Up) range: {min(z_coords):.1f} - {max(z_coords):.1f} m")

            # Count sources
            sources = {}
            for point in trajectory_data:
                source = point["source"]
                sources[source] = sources.get(source, 0) + 1

            print(f"\n📈 Data Sources:")
            for source, count in sources.items():
                percentage = (count / len(trajectory_data)) * 100
                print(f"   {source}: {count:,} points ({percentage:.1f}%)")

        print(f"\n✅ Test completed successfully!")
        print(
            f"   The JSON file is ready for use by TrajectoryTestActor in Unreal Engine."
        )

        return True

    except FileNotFoundError as e:
        print(f"❌ Flight files not found: {e}")
        print(
            f"   Make sure flight {flight_number} data exists in TrajectoryData directory"
        )
        return False
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def list_available_flights():
    """List available flights."""
    try:
        from flight_composer.flight_file import list_available_flights

        flights = list_available_flights()
        return flights
    except Exception as e:
        print(f"❌ Failed to list flights: {e}")
        return []


def main():
    """Main function."""
    print("Trajectory Export Test")
    print("=" * 50)

    # Check if flight number was provided
    if len(sys.argv) > 1:
        flight_number = sys.argv[1]
    else:
        # List available flights
        available_flights = list_available_flights()
        if not available_flights:
            print("❌ No flight files found!")
            print(
                "   Please ensure flight data exists in the TrajectoryData directory."
            )
            return

        flight_number = available_flights[0]
        print(f"Available flights: {available_flights}")
        print(
            f"Using flight {flight_number} (specify a different flight as command line argument)"
        )

    # Run the test
    success = test_trajectory_export(flight_number)

    if success:
        print(f"\n🎉 Ready for Unreal Engine!")
        print(f"   1. Open EPBC project in Unreal Engine")
        print(f"   2. Add TrajectoryTestActor to your level")
        print(f"   3. Set Flight Number to '{flight_number}'")
        print(f"   4. Click 'Load Trajectory Data' button")
        print(f"   5. Spheres should appear at trajectory points!")
    else:
        print(f"\n❌ Test failed - check error messages above")


if __name__ == "__main__":
    main()
