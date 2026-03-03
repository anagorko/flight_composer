"""
Unit tests for extract_telemetry_csv task.

This test module provides unit tests for the telemetry extraction functionality,
including finding GoPro MP4 files and extracting CSV data.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flight_composer import config
from flight_composer.flight_file import find_gopro_file, list_available_flights


class TestExtractTelemetryCSV:
    """Test telemetry CSV extraction functionality."""

    def test_find_gopro_file_case_insensitive(self):
        """Test that find_gopro_file works with both .mp4 and .MP4 extensions."""
        # This test checks the case-insensitive extension matching
        # We'll use a temporary directory to simulate the file structure

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different cases
            (temp_path / "03_test_flight.MP4").touch()
            (temp_path / "04_another_flight.mp4").touch()

            # Test finding MP4 file (uppercase extension)
            result = find_gopro_file("03", [temp_path])
            assert result is not None
            assert result.name == "03_test_flight.MP4"

            # Test finding mp4 file (lowercase extension)
            result = find_gopro_file("04", [temp_path])
            assert result is not None
            assert result.name == "04_another_flight.mp4"

            # Test non-existent flight
            result = find_gopro_file("99", [temp_path])
            assert result is None

    def test_find_gopro_file_with_flight_patterns(self):
        """Test that find_gopro_file works with different flight number patterns."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different flight number patterns
            (temp_path / "7_single_digit.mp4").touch()
            (temp_path / "07_zero_padded.MP4").touch()
            (temp_path / "007_triple_digit.mp4").touch()

            # Test finding with single digit
            result = find_gopro_file("7", [temp_path])
            assert result is not None
            assert "7" in result.name or "07" in result.name or "007" in result.name

            # Test finding with zero-padded
            result = find_gopro_file("07", [temp_path])
            assert result is not None
            assert "7" in result.name or "07" in result.name or "007" in result.name

    @patch("flight_composer.config.GOPRO_DIR", "/mock/gopro/dir")
    def test_find_gopro_file_default_dirs(self):
        """Test that find_gopro_file uses appropriate default directories."""

        with patch("pathlib.Path.exists") as mock_exists:
            # Mock GOPRO_DIR doesn't exist
            mock_exists.return_value = False

            with patch("flight_composer.flight_file.find_flight_file") as mock_find:
                mock_find.return_value = None

                result = find_gopro_file("03")

                # Should fall back to FLIGHT_SEARCH_DIRS
                assert mock_find.called
                call_args = mock_find.call_args_list
                assert len(call_args) == 2  # Called for both mp4 and MP4
                assert call_args[0][0][2] == config.FLIGHT_SEARCH_DIRS

    @pytest.mark.skipif(
        not os.path.exists(config.GOPRO_DIR) or not list_available_flights(),
        reason="No GoPro directory or flight data available for integration test",
    )
    def test_extract_telemetry_csv_integration(self):
        """Integration test for CSV extraction with flight 03 (if available)."""

        # Check if flight 03 exists
        flight_number = "03"
        available_flights = list_available_flights()

        # Try to find flight 03, or use the first available flight
        if flight_number not in available_flights:
            if not available_flights:
                pytest.skip("No flight data available for integration test")
            flight_number = available_flights[0]

        # Try to find the GoPro file
        gopro_file = find_gopro_file(flight_number)

        if gopro_file is None or not gopro_file.exists():
            pytest.skip(f"No GoPro MP4 file found for flight {flight_number}")

        # Test that we can at least find the file
        assert gopro_file.exists()
        assert gopro_file.suffix.lower() == ".mp4"

        # Check target CSV directory exists or can be created
        csv_dir = Path(config.TRAJECTORY_CSV_DIR)
        csv_dir.mkdir(parents=True, exist_ok=True)
        assert csv_dir.exists()

        # Generate expected CSV filename
        expected_csv_name = gopro_file.name.replace(".mp4", ".csv").replace(
            ".MP4", ".csv"
        )
        expected_csv_path = csv_dir / expected_csv_name

        # This integration test just verifies the file discovery works
        # The actual CSV extraction would require running the gopro-to-csv.py script
        logging.info(f"Would extract from {gopro_file} to {expected_csv_path}")

    def test_csv_path_generation(self):
        """Test CSV output path generation logic."""

        # Test with uppercase extension
        mp4_path = Path("/test/03_flight_data.MP4")
        expected_csv = "03_flight_data.csv"

        csv_name = mp4_path.name.replace(".mp4", ".csv").replace(".MP4", ".csv")
        assert csv_name == expected_csv

        # Test with lowercase extension
        mp4_path = Path("/test/03_flight_data.mp4")
        expected_csv = "03_flight_data.csv"

        csv_name = mp4_path.name.replace(".mp4", ".csv").replace(".MP4", ".csv")
        assert csv_name == expected_csv

    def test_file_modification_time_logic(self):
        """Test logic for comparing file modification times."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create MP4 file first
            mp4_file = temp_path / "03_test.mp4"
            mp4_file.touch()

            # Wait a moment and create CSV file (should be newer)
            import time

            time.sleep(0.1)

            csv_file = temp_path / "03_test.csv"
            csv_file.touch()

            # CSV should be newer than MP4
            assert csv_file.stat().st_mtime > mp4_file.stat().st_mtime

            # Now modify MP4 to be newer
            time.sleep(0.1)
            mp4_file.touch()

            # MP4 should now be newer than CSV
            assert mp4_file.stat().st_mtime > csv_file.stat().st_mtime

    def test_helper_script_exists(self):
        """Test that the helper script exists in the expected location."""
        script_path = (
            Path(__file__).parent.parent
            / "src"
            / "flight_composer"
            / "scripts"
            / "extract_telemetry_csv.py"
        )

        assert script_path.exists(), f"Helper script not found at {script_path}"

    def test_helper_script_imports(self):
        """Test that the helper script can be imported without errors."""
        import sys

        script_dir = (
            Path(__file__).parent.parent / "src" / "flight_composer" / "scripts"
        )
        sys.path.insert(0, str(script_dir))

        try:
            # Import the module to test basic functionality
            import extract_telemetry_csv as helper_module

            # Test that key functions exist
            assert hasattr(helper_module, "main")
            assert hasattr(helper_module, "extract_telemetry_for_flight")
            assert hasattr(helper_module, "should_process_flight")
            assert hasattr(helper_module, "generate_csv_path")

        finally:
            sys.path.pop(0)

    def test_direct_script_execution(self):
        """Test that the helper script can be executed directly."""
        import subprocess
        import sys

        script_path = (
            Path(__file__).parent.parent
            / "src"
            / "flight_composer"
            / "scripts"
            / "extract_telemetry_csv.py"
        )

        # Test help command
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Extract telemetry data from GoPro MP4 files" in result.stdout

        # Test with non-existent flight (should fail gracefully)
        result = subprocess.run(
            [sys.executable, str(script_path), "--flight-number", "999"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1  # Should fail
        assert "No MP4 file found for flight 999" in result.stderr
