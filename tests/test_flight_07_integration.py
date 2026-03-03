"""
Integration test using actual flight 07 data files.

This test attempts to use real flight 07 data files if they exist in the
configured directories, providing a real-world test of the Phase I pipeline.
"""

import logging
from pathlib import Path

import pytest

from flight_composer.flight_file import find_flight_file
from flight_composer.trajectory import config
from flight_composer.trajectory.ingestion import process_flight_phase_i


class TestFlight07Integration:
    """Test Phase I pipeline with actual flight 07 data."""

    def test_flight_07_files_exist(self):
        """Check if flight 07 files exist in configured directories."""
        extensions = ["igc", "gpx", "geojson"]
        available_files = {}

        for ext in extensions:
            file_path = find_flight_file("07", ext)
            if file_path and file_path.exists():
                available_files[ext] = file_path

        # At least one file should exist for meaningful test
        if not available_files:
            pytest.skip("No flight 07 data files found in configured directories")

        print(f"\nFound flight 07 files: {list(available_files.keys())}")
        for ext, path in available_files.items():
            print(f"  {ext}: {path}")

    @pytest.mark.skipif(
        not any(
            find_flight_file("07", ext) and find_flight_file("07", ext).exists()
            for ext in ["igc", "gpx", "geojson"]
        ),
        reason="No flight 07 data files available",
    )
    def test_process_flight_07_phase_i(self):
        """Test complete Phase I processing with flight 07 data."""
        # Enable debug logging for this test
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()

        try:
            # Process flight 07 through Phase I
            result = process_flight_phase_i(flight_number="07")

            # Verify we got valid results
            assert result is not None
            assert len(result.trajectory_points) > 0
            assert result.metadata is not None
            assert result.metadata.flight_number == "07"
            assert result.metadata.origin is not None

            # Check trajectory properties
            assert result.duration > 0
            assert result.start_time is not None
            assert result.end_time is not None
            assert result.start_time < result.end_time

            # Check coordinate origin is reasonable (should be somewhere in Europe/Poland area)
            origin = result.metadata.origin
            assert 50.0 <= origin.latitude <= 55.0  # Rough Poland latitude range
            assert 14.0 <= origin.longitude <= 25.0  # Rough Poland longitude range

            # Get trajectory arrays for analysis
            timestamps, x_coords, y_coords, z_coords = result.get_trajectory_arrays()

            assert len(timestamps) > 10  # Should have reasonable number of points
            # Check for reasonable movement in trajectory
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            assert x_range > 100  # Should have at least 100m movement in x
            assert y_range > 100  # Should have at least 100m movement in y

            # Log summary information
            print(f"\nFlight 07 Phase I processing results:")
            print(f"  Duration: {result.duration:.1f} seconds")
            print(f"  Trajectory points: {len(result.trajectory_points)}")
            print(f"  Start time: {result.start_time}")
            print(f"  End time: {result.end_time}")
            print(
                f"  Origin: lat={origin.latitude:.6f}, lon={origin.longitude:.6f}, alt={origin.altitude_msl:.1f}m"
            )
            print(
                f"  Coordinate bounds: x=[{min(x_coords):.1f}, {max(x_coords):.1f}], "
                f"y=[{min(y_coords):.1f}, {max(y_coords):.1f}], z=[{min(z_coords):.1f}, {max(z_coords):.1f}]"
            )

            if result.processing_notes:
                print("  Processing notes:")
                for note in result.processing_notes:
                    print(f"    - {note}")

            # Test trajectory array extraction
            assert len(timestamps) == len(x_coords) == len(y_coords) == len(z_coords)

        except Exception as e:
            pytest.fail(f"Phase I processing failed for flight 07: {e}")

    def test_flight_07_data_validation(self):
        """Test validation of flight 07 data files."""
        extensions = ["igc", "gpx", "geojson"]

        for ext in extensions:
            file_path = find_flight_file("07", ext)
            if file_path and file_path.exists():
                print(f"\nValidating {ext} file: {file_path}")

                # Check file size
                file_size = file_path.stat().st_size
                assert file_size > 0, f"{ext} file is empty"
                print(f"  File size: {file_size} bytes")

                # Check file is readable
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline()
                        assert len(first_line) > 0, f"{ext} file appears to be empty"
                        print(f"  First line: {first_line[:50]}...")

                        # Count total lines for reference
                        f.seek(0)
                        line_count = sum(1 for _ in f)
                        print(f"  Total lines: {line_count}")

                except Exception as e:
                    pytest.fail(f"Could not read {ext} file: {e}")

    def test_configured_directories_exist(self):
        """Test that configured search directories exist."""
        search_dirs = config.FLIGHT_SEARCH_DIRS

        print(f"\nConfigured search directories:")
        for search_dir in search_dirs:
            print(f"  {search_dir}")
            if search_dir.exists():
                print(f"    ✓ Exists")

                # Count flight files in directory
                flight_files = []
                for pattern in ["*_*.igc", "*_*.gpx", "*_*.geojson"]:
                    flight_files.extend(search_dir.glob(pattern))

                print(f"    Found {len(flight_files)} flight files")

                # List first few files as examples
                for i, file_path in enumerate(flight_files[:3]):
                    print(f"      {file_path.name}")
                if len(flight_files) > 3:
                    print(f"      ... and {len(flight_files) - 3} more")

            else:
                print(f"    ✗ Does not exist")

        # At least one directory should exist
        existing_dirs = [d for d in search_dirs if d.exists()]
        assert len(existing_dirs) > 0, "No configured search directories exist"
