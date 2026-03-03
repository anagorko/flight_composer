"""
Integration tests for Phase I flight data ingestion pipeline.

These tests verify the complete Phase I processing pipeline from raw flight
files to NormalizedFlightData, using test data files for flight "07".
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from flight_composer.trajectory.data_structures import (
    NormalizedFlightData,
    RawFlightPoint,
    TrajectoryPoint,
)
from flight_composer.trajectory.ingestion import (
    FlightDataIngester,
    process_flight_phase_i,
)


class TestFlightDataIngestionIntegration:
    """Integration tests for complete Phase I pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ingester = FlightDataIngester()

    @pytest.fixture
    def sample_igc_content(self):
        """Sample IGC file content for testing."""
        return """HFDTE020521
HFPLT:Test Pilot
HFGTY:Test Glider
B1000005200000N02000000EA0010000100
B1001005201000N02001000EA0010100110
B1002005202000N02002000EA0010200120
B1003005203000N02003000EA0010300130
B1004005204000N02004000EA0010400140
B1005005205000N02005000EA0010300130
"""

    @pytest.fixture
    def sample_gpx_content(self):
        """Sample GPX file content for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
<trk>
<trkseg>
<trkpt lat="52.0" lon="20.0">
<ele>100</ele>
<time>2021-05-02T10:00:05Z</time>
<speed>5.0</speed>
<hdop>2.0</hdop>
</trkpt>
<trkpt lat="52.01" lon="20.01">
<ele>110</ele>
<time>2021-05-02T10:01:05Z</time>
<speed>15.0</speed>
<hdop>2.5</hdop>
</trkpt>
<trkpt lat="52.02" lon="20.02">
<ele>120</ele>
<time>2021-05-02T10:02:05Z</time>
<speed>25.0</speed>
<hdop>1.8</hdop>
</trkpt>
<trkpt lat="52.03" lon="20.03">
<ele>130</ele>
<time>2021-05-02T10:03:05Z</time>
<speed>20.0</speed>
<hdop>2.2</hdop>
</trkpt>
</trkseg>
</trk>
</gpx>"""

    @pytest.fixture
    def sample_geojson_content(self):
        """Sample GeoJSON file content for testing."""
        return {
            "type": "Feature",
            "properties": {"target_speed": 25.0},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [20.0, 52.0, 100],
                    [20.01, 52.01, 110],
                    [20.02, 52.02, 120],
                    [20.03, 52.03, 130],
                    [20.04, 52.04, 140],
                ],
            },
        }

    @pytest.fixture
    def temp_flight_files(
        self, sample_igc_content, sample_gpx_content, sample_geojson_content
    ):
        """Create temporary flight files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create IGC file
            igc_file = temp_path / "07_test_flight.igc"
            igc_file.write_text(sample_igc_content)

            # Create GPX file
            gpx_file = temp_path / "07_test_flight.gpx"
            gpx_file.write_text(sample_gpx_content)

            # Create GeoJSON file
            geojson_file = temp_path / "07_test_flight.geojson"
            geojson_file.write_text(json.dumps(sample_geojson_content))

            yield {
                "temp_dir": temp_path,
                "igc_file": igc_file,
                "gpx_file": gpx_file,
                "geojson_file": geojson_file,
            }

    def test_discover_flight_files(self, temp_flight_files):
        """Test flight file discovery functionality."""
        temp_dir = temp_flight_files["temp_dir"]

        # Mock find_flight_file to return our test files
        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                file_map = {
                    "igc": temp_flight_files["igc_file"],
                    "gpx": temp_flight_files["gpx_file"],
                    "geojson": temp_flight_files["geojson_file"],
                }
                return file_map.get(ext)

            mock_find.side_effect = mock_find_impl

            available_files = self.ingester._discover_flight_files("07", [temp_dir])

            assert "igc" in available_files
            assert "gpx" in available_files
            assert "geojson" in available_files
            assert len(available_files) == 3

    def test_load_raw_data_igc_only(self, temp_flight_files):
        """Test raw data loading with IGC file only."""
        file_paths = {"igc": temp_flight_files["igc_file"]}

        raw_sources = self.ingester._load_raw_data(file_paths)

        assert "igc" in raw_sources
        assert len(raw_sources["igc"]) == 6  # 6 B records in sample IGC

        # Check first point
        first_point = raw_sources["igc"][0]
        assert isinstance(first_point, RawFlightPoint)
        assert first_point.source == "IGC"
        assert abs(first_point.latitude - 52.0) < 0.001
        assert abs(first_point.longitude - 20.0) < 0.001

    def test_load_raw_data_gpx_only(self, temp_flight_files):
        """Test raw data loading with GPX file only."""
        file_paths = {"gpx": temp_flight_files["gpx_file"]}

        raw_sources = self.ingester._load_raw_data(file_paths)

        assert "gpx" in raw_sources
        assert len(raw_sources["gpx"]) == 4  # 4 track points in sample GPX

        # Check first point
        first_point = raw_sources["gpx"][0]
        assert isinstance(first_point, RawFlightPoint)
        assert first_point.source == "GPX"
        assert first_point.hdop == 2.0
        assert first_point.speed == 5.0

    def test_load_raw_data_geojson_only(self, temp_flight_files):
        """Test raw data loading with GeoJSON file only."""
        file_paths = {"geojson": temp_flight_files["geojson_file"]}

        raw_sources = self.ingester._load_raw_data(file_paths)

        assert "geojson" in raw_sources
        assert len(raw_sources["geojson"]) == 5  # 5 coordinates in LineString

        # Check first point
        first_point = raw_sources["geojson"][0]
        assert isinstance(first_point, RawFlightPoint)
        assert first_point.source == "GEOJSON"
        assert abs(first_point.latitude - 52.0) < 0.001
        assert abs(first_point.longitude - 20.0) < 0.001

    def test_load_raw_data_all_sources(self, temp_flight_files):
        """Test raw data loading with all file types."""
        file_paths = {
            "igc": temp_flight_files["igc_file"],
            "gpx": temp_flight_files["gpx_file"],
            "geojson": temp_flight_files["geojson_file"],
        }

        raw_sources = self.ingester._load_raw_data(file_paths)

        assert "igc" in raw_sources
        assert "gpx" in raw_sources
        assert "geojson" in raw_sources
        assert len(raw_sources) == 3

    def test_clean_data_sources(self, temp_flight_files):
        """Test data cleaning pipeline."""
        file_paths = {"igc": temp_flight_files["igc_file"]}
        raw_sources = self.ingester._load_raw_data(file_paths)

        cleaned_sources = self.ingester._clean_data_sources(raw_sources)

        assert "igc" in cleaned_sources
        assert len(cleaned_sources["igc"]) > 0

        # Check that points are time-sorted
        points = cleaned_sources["igc"]
        for i in range(1, len(points)):
            assert points[i].timestamp >= points[i - 1].timestamp

    def test_coordinate_transformation(self, temp_flight_files):
        """Test coordinate system transformation."""
        file_paths = {"igc": temp_flight_files["igc_file"]}
        raw_sources = self.ingester._load_raw_data(file_paths)
        cleaned_sources = self.ingester._clean_data_sources(raw_sources)

        # Determine origin
        origin = self.ingester._determine_coordinate_origin(cleaned_sources)
        assert abs(origin.latitude - 52.0) < 0.02
        assert abs(origin.longitude - 20.0) < 0.02

        # Transform to local coordinates
        local_sources = self.ingester._transform_to_local_coordinates(
            cleaned_sources, origin
        )

        assert "igc" in local_sources
        points = local_sources["igc"]
        assert len(points) > 0

        # Check first point should be near origin (0, 0, 0)
        first_point = points[0]
        assert isinstance(first_point, TrajectoryPoint)
        assert abs(first_point.x) < 10.0  # Should be near zero
        assert abs(first_point.y) < 10.0  # Should be near zero

    def test_align_and_merge_geojson_priority(self, temp_flight_files):
        """Test data alignment with GeoJSON priority."""
        file_paths = {
            "geojson": temp_flight_files["geojson_file"],
            "igc": temp_flight_files["igc_file"],
        }
        raw_sources = self.ingester._load_raw_data(file_paths)
        cleaned_sources = self.ingester._clean_data_sources(raw_sources)
        origin = self.ingester._determine_coordinate_origin(cleaned_sources)
        local_sources = self.ingester._transform_to_local_coordinates(
            cleaned_sources, origin
        )

        # GeoJSON should have platinum priority
        aligned_trajectory = self.ingester._align_and_merge_sources(local_sources)

        assert len(aligned_trajectory) > 0
        # Should use GeoJSON data (all points from same source)
        assert all(p.source == "GEOJSON" for p in aligned_trajectory)

    def test_align_and_merge_igc_gpx(self, temp_flight_files):
        """Test data alignment between IGC and GPX sources."""
        file_paths = {
            "igc": temp_flight_files["igc_file"],
            "gpx": temp_flight_files["gpx_file"],
        }
        raw_sources = self.ingester._load_raw_data(file_paths)
        cleaned_sources = self.ingester._clean_data_sources(raw_sources)
        origin = self.ingester._determine_coordinate_origin(cleaned_sources)
        local_sources = self.ingester._transform_to_local_coordinates(
            cleaned_sources, origin
        )

        aligned_trajectory = self.ingester._align_and_merge_sources(local_sources)

        assert len(aligned_trajectory) > 0
        # Should have merged trajectory with IGC as primary
        assert any("igc" in p.source.lower() for p in aligned_trajectory)

    def test_complete_pipeline_integration(self, temp_flight_files):
        """Test complete Phase I pipeline integration."""
        temp_dir = temp_flight_files["temp_dir"]

        # Mock find_flight_file to return our test files
        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                file_map = {
                    "igc": temp_flight_files["igc_file"],
                    "gpx": temp_flight_files["gpx_file"],
                    "geojson": temp_flight_files["geojson_file"],
                }
                return file_map.get(ext)

            mock_find.side_effect = mock_find_impl

            # Process complete flight
            result = self.ingester.process_flight(
                flight_number="07", search_dirs=[temp_dir]
            )

            # Verify result type and structure
            assert isinstance(result, NormalizedFlightData)
            assert len(result.trajectory_points) > 0
            assert result.metadata is not None
            assert result.metadata.flight_number == "07"
            assert result.metadata.origin is not None

            # Verify trajectory properties
            assert result.duration > 0
            assert result.start_time is not None
            assert result.end_time is not None
            assert result.start_time < result.end_time

            # Verify processing notes
            assert len(result.processing_notes) > 0
            assert any("files:" in note for note in result.processing_notes)

    def test_process_flight_phase_i_convenience_function(self, temp_flight_files):
        """Test the convenience function for Phase I processing."""
        temp_dir = temp_flight_files["temp_dir"]

        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                file_map = {
                    "igc": temp_flight_files["igc_file"],
                }
                return file_map.get(ext)

            mock_find.side_effect = mock_find_impl

            # Test convenience function
            result = process_flight_phase_i(flight_number="07", search_dirs=[temp_dir])

            assert isinstance(result, NormalizedFlightData)
            assert len(result.trajectory_points) > 0

    def test_error_handling_no_files_found(self):
        """Test error handling when no flight files are found."""
        with patch("flight_composer.flight_file.find_flight_file", return_value=None):
            with pytest.raises(FileNotFoundError, match="No flight files found"):
                self.ingester.process_flight("99")

    def test_error_handling_invalid_file_content(self, temp_flight_files):
        """Test error handling with invalid file content."""
        temp_dir = temp_flight_files["temp_dir"]

        # Create invalid IGC file
        invalid_igc = temp_dir / "99_invalid.igc"
        invalid_igc.write_text("Invalid IGC content")

        with patch("flight_composer.flight_file.find_flight_file") as mock_find:
            mock_find.return_value = invalid_igc

            # Should handle invalid content gracefully
            with pytest.raises(ValueError, match="No valid data"):
                self.ingester.process_flight("99")

    def test_data_prioritization(self, temp_flight_files):
        """Test that data prioritization works correctly."""
        file_paths = {
            "igc": temp_flight_files["igc_file"],
            "gpx": temp_flight_files["gpx_file"],
            "geojson": temp_flight_files["geojson_file"],
        }

        raw_sources = self.ingester._load_raw_data(file_paths)
        cleaned_sources = self.ingester._clean_data_sources(raw_sources)
        origin = self.ingester._determine_coordinate_origin(cleaned_sources)
        local_sources = self.ingester._transform_to_local_coordinates(
            cleaned_sources, origin
        )

        # With all sources available, GeoJSON should have priority
        aligned_trajectory = self.ingester._align_and_merge_sources(local_sources)

        # Should use GeoJSON (platinum priority)
        assert all(p.source == "GEOJSON" for p in aligned_trajectory)

        # Test without GeoJSON - should use IGC as primary
        local_sources_no_geojson = {
            k: v for k, v in local_sources.items() if k != "geojson"
        }
        aligned_trajectory_no_geojson = self.ingester._align_and_merge_sources(
            local_sources_no_geojson
        )

        # Should have IGC-based trajectory (possibly merged with GPX)
        assert any("igc" in p.source.lower() for p in aligned_trajectory_no_geojson)

    def test_gap_processing(self, temp_flight_files):
        """Test gap detection and processing."""
        # Create IGC with time gaps
        igc_with_gaps = """HFDTE020521
HFPLT:Test Pilot
B1000005200000N02000000EA0010000100
B1001005201000N02001000EA0010100110
B1010005202000N02002000EA0010200120
B1011005203000N02003000EA0010300130
"""

        temp_dir = temp_flight_files["temp_dir"]
        igc_gap_file = temp_dir / "07_gaps.igc"
        igc_gap_file.write_text(igc_with_gaps)

        file_paths = {"igc": igc_gap_file}
        raw_sources = self.ingester._load_raw_data(file_paths)
        cleaned_sources = self.ingester._clean_data_sources(raw_sources)
        origin = self.ingester._determine_coordinate_origin(cleaned_sources)
        local_sources = self.ingester._transform_to_local_coordinates(
            cleaned_sources, origin
        )
        aligned_trajectory = self.ingester._align_and_merge_sources(local_sources)

        # Process gaps
        processed_trajectory = self.ingester._process_gaps(aligned_trajectory)

        assert len(processed_trajectory) >= len(
            aligned_trajectory
        )  # May fill micro gaps

    def test_trajectory_arrays_extraction(self, temp_flight_files):
        """Test extraction of trajectory data as numpy arrays."""
        temp_dir = temp_flight_files["temp_dir"]

        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                return temp_flight_files["igc_file"] if ext == "igc" else None

            mock_find.side_effect = mock_find_impl

            result = self.ingester.process_flight("07", search_dirs=[temp_dir])

            timestamps, x_coords, y_coords, z_coords = result.get_trajectory_arrays()

            assert len(timestamps) > 0
            assert len(x_coords) == len(timestamps)
            assert len(y_coords) == len(timestamps)
            assert len(z_coords) == len(timestamps)

            # Check that arrays are properly sorted by time
            assert np.all(np.diff(timestamps) >= 0)  # Should be monotonic

    def test_metadata_creation(self, temp_flight_files):
        """Test flight metadata creation."""
        temp_dir = temp_flight_files["temp_dir"]

        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                return temp_flight_files["igc_file"] if ext == "igc" else None

            mock_find.side_effect = mock_find_impl

            result = self.ingester.process_flight(
                flight_number="07",
                search_dirs=[temp_dir],
                glider_polar={"type": "test_glider"},
                wind_vector=(5.0, -2.0, 0.1),
                terrain_offset=15.0,
            )

            metadata = result.metadata
            assert metadata.flight_number == "07"
            assert metadata.origin is not None
            assert metadata.glider_polar == {"type": "test_glider"}
            assert metadata.wind_vector == (5.0, -2.0, 0.1)
            assert metadata.terrain_offset == 15.0

    def test_processing_notes_generation(self, temp_flight_files):
        """Test that processing notes are properly generated."""
        temp_dir = temp_flight_files["temp_dir"]

        with patch("flight_composer.flight_file.find_flight_file") as mock_find:

            def mock_find_impl(flight_num, ext, search_dirs=None):
                file_map = {
                    "igc": temp_flight_files["igc_file"],
                    "gpx": temp_flight_files["gpx_file"],
                }
                return file_map.get(ext)

            mock_find.side_effect = mock_find_impl

            result = self.ingester.process_flight("07", search_dirs=[temp_dir])

            # Check processing notes
            notes = result.processing_notes
            assert len(notes) > 0

            # Should have notes about processed files
            file_note = next((note for note in notes if "files:" in note), None)
            assert file_note is not None
            assert "igc" in file_note
            assert "gpx" in file_note

            # Should have note about final trajectory
            trajectory_note = next(
                (note for note in notes if "Final trajectory:" in note), None
            )
            assert trajectory_note is not None
