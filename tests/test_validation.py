"""
Unit tests for flight data validation module.

Tests the validation and cleaning functions used in Phase I data processing,
including flight window detection, data quality checks, and cleaning operations.
"""

import numpy as np
import pytest

from flight_composer.trajectory.data_structures import RawFlightPoint
from flight_composer.trajectory.validation import (
    calculate_ground_speed,
    clean_flight_data,
    detect_altitude_spikes,
    detect_flight_window,
    identify_data_gaps,
    prune_stationary_points,
    validate_coordinate_bounds,
    validate_gps_quality,
    validate_temporal_consistency,
)


class TestDetectFlightWindow:
    """Test flight window detection functionality."""

    def test_detect_flight_window_basic(self):
        """Test basic flight window detection."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=1.0,
            ),  # Stationary
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=2.0,
            ),  # Still slow
            RawFlightPoint(
                timestamp=1020.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                speed=15.0,
            ),  # Takeoff speed
            RawFlightPoint(
                timestamp=1030.0,
                latitude=52.2,
                longitude=20.2,
                altitude_msl=120.0,
                speed=20.0,
            ),  # Flying
            RawFlightPoint(
                timestamp=1040.0,
                latitude=52.3,
                longitude=20.3,
                altitude_msl=130.0,
                speed=25.0,
            ),  # Flying
            RawFlightPoint(
                timestamp=1050.0,
                latitude=52.4,
                longitude=20.4,
                altitude_msl=140.0,
                speed=22.0,
            ),  # Flying
            RawFlightPoint(
                timestamp=1060.0,
                latitude=52.5,
                longitude=20.5,
                altitude_msl=120.0,
                speed=18.0,
            ),  # Landing approach
            RawFlightPoint(
                timestamp=1070.0,
                latitude=52.6,
                longitude=20.6,
                altitude_msl=105.0,
                speed=12.0,
            ),  # Landing
            RawFlightPoint(
                timestamp=1080.0,
                latitude=52.6,
                longitude=20.6,
                altitude_msl=100.0,
                speed=1.0,
            ),  # Landed
        ]

        takeoff_time, landing_time = detect_flight_window(
            points, min_speed_kmh=10.0, sustain_duration=5.0
        )

        assert takeoff_time == 1020.0  # First sustained high speed
        assert landing_time == 1070.0  # Last high speed point

    def test_detect_flight_window_no_speed_data(self):
        """Test flight window detection with no speed data."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),  # No speed
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),  # No speed
        ]

        takeoff_time, landing_time = detect_flight_window(points)

        assert takeoff_time is None
        assert landing_time is None

    def test_detect_flight_window_empty_points(self):
        """Test flight window detection with empty point list."""
        takeoff_time, landing_time = detect_flight_window([])

        assert takeoff_time is None
        assert landing_time is None

    def test_detect_flight_window_no_sustained_flight(self):
        """Test flight window detection where no sustained flight is detected."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=4.0,
            ),  # Brief high speed (above 2.78 m/s threshold)
            RawFlightPoint(
                timestamp=1002.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=1.0,
            ),  # Back to slow (drops before 5s sustain)
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=2.0,
            ),  # Still slow
        ]

        takeoff_time, landing_time = detect_flight_window(points, sustain_duration=5.0)

        assert (
            takeoff_time is None
        )  # No sustained flight (speed dropped at 1002.0, before 1005.0)
        assert landing_time == 1000.0  # Last high speed point


class TestPruneStationaryPoints:
    """Test stationary point pruning functionality."""

    def test_prune_stationary_points_with_window(self):
        """Test pruning with explicit flight window."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),
            RawFlightPoint(
                timestamp=1020.0, latitude=52.2, longitude=20.2, altitude_msl=120.0
            ),
            RawFlightPoint(
                timestamp=1030.0, latitude=52.3, longitude=20.3, altitude_msl=130.0
            ),
            RawFlightPoint(
                timestamp=1040.0, latitude=52.4, longitude=20.4, altitude_msl=140.0
            ),
        ]

        filtered_points = prune_stationary_points(
            points, takeoff_time=1010.0, landing_time=1030.0
        )

        assert len(filtered_points) == 3
        assert filtered_points[0].timestamp == 1010.0
        assert filtered_points[-1].timestamp == 1030.0

    def test_prune_stationary_points_auto_detect(self):
        """Test pruning with auto-detection of flight window."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=1.0,
            ),
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                speed=15.0,
            ),
            RawFlightPoint(
                timestamp=1020.0,
                latitude=52.2,
                longitude=20.2,
                altitude_msl=120.0,
                speed=20.0,
            ),
            RawFlightPoint(
                timestamp=1030.0,
                latitude=52.3,
                longitude=20.3,
                altitude_msl=130.0,
                speed=2.0,
            ),
        ]

        filtered_points = prune_stationary_points(points)

        # Should detect flight window and filter accordingly
        assert len(filtered_points) <= len(points)

    def test_prune_stationary_points_empty(self):
        """Test pruning with empty point list."""
        filtered_points = prune_stationary_points([])
        assert filtered_points == []


class TestValidateGpsQuality:
    """Test GPS quality validation functionality."""

    def test_validate_gps_quality_basic(self):
        """Test basic GPS quality validation."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                hdop=2.0,
            ),  # Good
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                hdop=8.0,
            ),  # Poor
            RawFlightPoint(
                timestamp=1020.0,
                latitude=52.2,
                longitude=20.2,
                altitude_msl=120.0,
                hdop=3.0,
            ),  # Good
            RawFlightPoint(
                timestamp=1030.0,
                latitude=52.3,
                longitude=20.3,
                altitude_msl=130.0,
                hdop=None,
            ),  # No HDOP
        ]

        filtered_points = validate_gps_quality(points, max_hdop=5.0)

        assert len(filtered_points) == 3  # One point rejected due to high HDOP
        assert all(p.hdop is None or p.hdop <= 5.0 for p in filtered_points)

    def test_validate_gps_quality_all_good(self):
        """Test GPS quality validation with all good points."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                hdop=2.0,
            ),
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                hdop=3.0,
            ),
        ]

        filtered_points = validate_gps_quality(points, max_hdop=5.0)

        assert len(filtered_points) == len(points)

    def test_validate_gps_quality_empty(self):
        """Test GPS quality validation with empty point list."""
        filtered_points = validate_gps_quality([])
        assert filtered_points == []


class TestDetectAltitudeSpikes:
    """Test altitude spike detection functionality."""

    def test_detect_altitude_spikes_basic(self):
        """Test basic altitude spike detection."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1001.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),  # Normal climb
            RawFlightPoint(
                timestamp=1002.0, latitude=52.2, longitude=20.2, altitude_msl=200.0
            ),  # Impossible spike
            RawFlightPoint(
                timestamp=1003.0, latitude=52.3, longitude=20.3, altitude_msl=120.0
            ),  # Normal
        ]

        filtered_points = detect_altitude_spikes(points, max_vertical_speed=20.0)

        assert len(filtered_points) == 3  # Spike removed
        assert 200.0 not in [p.altitude_msl for p in filtered_points]

    def test_detect_altitude_spikes_normal_data(self):
        """Test altitude spike detection with normal data."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),
            RawFlightPoint(
                timestamp=1020.0, latitude=52.2, longitude=20.2, altitude_msl=115.0
            ),
        ]

        filtered_points = detect_altitude_spikes(points, max_vertical_speed=20.0)

        assert len(filtered_points) == len(points)

    def test_detect_altitude_spikes_single_point(self):
        """Test altitude spike detection with single point."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
        ]

        filtered_points = detect_altitude_spikes(points)

        assert len(filtered_points) == 1


class TestValidateTemporalConsistency:
    """Test temporal consistency validation functionality."""

    def test_validate_temporal_consistency_basic(self):
        """Test basic temporal consistency validation."""
        points = [
            RawFlightPoint(
                timestamp=1020.0, latitude=52.2, longitude=20.2, altitude_msl=120.0
            ),  # Out of order
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=111.0
            ),  # Duplicate timestamp
        ]

        filtered_points = validate_temporal_consistency(points)

        assert len(filtered_points) == 3  # One duplicate removed
        assert (
            filtered_points[0].timestamp
            < filtered_points[1].timestamp
            < filtered_points[2].timestamp
        )

    def test_validate_temporal_consistency_already_sorted(self):
        """Test temporal consistency with already sorted data."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),
            RawFlightPoint(
                timestamp=1020.0, latitude=52.2, longitude=20.2, altitude_msl=120.0
            ),
        ]

        filtered_points = validate_temporal_consistency(points)

        assert len(filtered_points) == len(points)
        assert filtered_points == points

    def test_validate_temporal_consistency_empty(self):
        """Test temporal consistency with empty list."""
        filtered_points = validate_temporal_consistency([])
        assert filtered_points == []


class TestCalculateGroundSpeed:
    """Test ground speed calculation functionality."""

    def test_calculate_ground_speed_basic(self):
        """Test basic ground speed calculation."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=52.001, longitude=20.001, altitude_msl=110.0
            ),  # Small movement
        ]

        calculated_points = calculate_ground_speed(points)

        assert len(calculated_points) == len(points)
        assert calculated_points[0].speed is None  # First point has no previous point
        assert (
            calculated_points[1].speed is not None
        )  # Second point has calculated speed
        assert calculated_points[1].speed > 0  # Speed should be positive

    def test_calculate_ground_speed_existing_speeds(self):
        """Test ground speed calculation with existing speeds."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=25.0,
            ),
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.001,
                longitude=20.001,
                altitude_msl=110.0,
                speed=30.0,
            ),
        ]

        calculated_points = calculate_ground_speed(points)

        # Existing speeds should be preserved
        assert calculated_points[0].speed == 25.0
        assert calculated_points[1].speed == 30.0

    def test_calculate_ground_speed_single_point(self):
        """Test ground speed calculation with single point."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
        ]

        calculated_points = calculate_ground_speed(points)

        assert len(calculated_points) == 1
        assert calculated_points[0].speed is None


class TestIdentifyDataGaps:
    """Test data gap identification functionality."""

    def test_identify_data_gaps_basic(self):
        """Test basic data gap identification."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1001.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),  # 1s gap
            RawFlightPoint(
                timestamp=1010.0, latitude=52.2, longitude=20.2, altitude_msl=120.0
            ),  # 9s gap
            RawFlightPoint(
                timestamp=1011.0, latitude=52.3, longitude=20.3, altitude_msl=130.0
            ),  # 1s gap
        ]

        gaps = identify_data_gaps(points, gap_threshold=5.0)

        assert len(gaps) == 1  # Only one gap > 5s
        assert gaps[0] == (1001.0, 1010.0)

    def test_identify_data_gaps_no_gaps(self):
        """Test data gap identification with no gaps."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1001.0, latitude=52.1, longitude=20.1, altitude_msl=110.0
            ),
            RawFlightPoint(
                timestamp=1002.0, latitude=52.2, longitude=20.2, altitude_msl=120.0
            ),
        ]

        gaps = identify_data_gaps(points, gap_threshold=5.0)

        assert len(gaps) == 0

    def test_identify_data_gaps_empty(self):
        """Test data gap identification with empty list."""
        gaps = identify_data_gaps([])
        assert gaps == []


class TestValidateCoordinateBounds:
    """Test coordinate bounds validation functionality."""

    def test_validate_coordinate_bounds_basic(self):
        """Test basic coordinate bounds validation."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),  # Valid
            RawFlightPoint(
                timestamp=1010.0, latitude=95.0, longitude=20.0, altitude_msl=110.0
            ),  # Invalid lat
            RawFlightPoint(
                timestamp=1020.0, latitude=52.0, longitude=185.0, altitude_msl=120.0
            ),  # Invalid lon
            RawFlightPoint(
                timestamp=1030.0, latitude=52.0, longitude=20.0, altitude_msl=25000.0
            ),  # Invalid alt
            RawFlightPoint(
                timestamp=1040.0, latitude=53.0, longitude=21.0, altitude_msl=150.0
            ),  # Valid
        ]

        filtered_points = validate_coordinate_bounds(points)

        assert len(filtered_points) == 2  # Only valid points remain
        assert all(p.latitude <= 90.0 for p in filtered_points)
        assert all(p.longitude <= 180.0 for p in filtered_points)
        assert all(p.altitude_msl <= 20000.0 for p in filtered_points)

    def test_validate_coordinate_bounds_all_valid(self):
        """Test coordinate bounds validation with all valid points."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=53.0, longitude=21.0, altitude_msl=110.0
            ),
        ]

        filtered_points = validate_coordinate_bounds(points)

        assert len(filtered_points) == len(points)

    def test_validate_coordinate_bounds_custom_bounds(self):
        """Test coordinate bounds validation with custom bounds."""
        points = [
            RawFlightPoint(
                timestamp=1000.0, latitude=52.0, longitude=20.0, altitude_msl=100.0
            ),
            RawFlightPoint(
                timestamp=1010.0, latitude=53.0, longitude=21.0, altitude_msl=500.0
            ),
        ]

        # Very restrictive altitude bounds
        filtered_points = validate_coordinate_bounds(points, max_alt=200.0)

        assert len(filtered_points) == 1  # Second point rejected due to altitude


class TestCleanFlightData:
    """Test complete flight data cleaning pipeline."""

    def test_clean_flight_data_complete_pipeline(self):
        """Test complete data cleaning pipeline."""
        points = [
            # Pre-flight stationary points
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=2.0,
            ),
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=3.0,
            ),
            # Flight data with some issues
            RawFlightPoint(
                timestamp=1020.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                speed=15.0,
                hdop=2.0,
            ),
            RawFlightPoint(
                timestamp=1025.0,
                latitude=52.15,
                longitude=20.15,
                altitude_msl=400.0,
                hdop=3.0,
            ),  # Altitude spike
            RawFlightPoint(
                timestamp=1030.0,
                latitude=52.2,
                longitude=20.2,
                altitude_msl=120.0,
                speed=20.0,
                hdop=8.0,
            ),  # Poor GPS
            RawFlightPoint(
                timestamp=1040.0,
                latitude=52.3,
                longitude=20.3,
                altitude_msl=130.0,
                speed=25.0,
                hdop=2.0,
            ),
            RawFlightPoint(
                timestamp=1050.0,
                latitude=52.4,
                longitude=20.4,
                altitude_msl=125.0,
                speed=18.0,
                hdop=3.0,
            ),
            # Landing and post-flight
            RawFlightPoint(
                timestamp=1060.0,
                latitude=52.5,
                longitude=20.5,
                altitude_msl=105.0,
                speed=8.0,
            ),
            RawFlightPoint(
                timestamp=1070.0,
                latitude=52.5,
                longitude=20.5,
                altitude_msl=100.0,
                speed=1.0,
            ),
        ]

        cleaned_points, takeoff_time, landing_time = clean_flight_data(points)

        # Should remove stationary points, altitude spikes, and poor GPS quality points
        assert len(cleaned_points) < len(points)
        assert takeoff_time is not None
        assert landing_time is not None

        # All remaining points should have good GPS quality
        for point in cleaned_points:
            if point.hdop is not None:
                assert point.hdop <= 5.0

    def test_clean_flight_data_empty(self):
        """Test data cleaning with empty input."""
        cleaned_points, takeoff_time, landing_time = clean_flight_data([])

        assert cleaned_points == []
        assert takeoff_time is None
        assert landing_time is None

    def test_clean_flight_data_minimal_data(self):
        """Test data cleaning with minimal valid data."""
        points = [
            RawFlightPoint(
                timestamp=1000.0,
                latitude=52.0,
                longitude=20.0,
                altitude_msl=100.0,
                speed=15.0,
            ),
            RawFlightPoint(
                timestamp=1010.0,
                latitude=52.1,
                longitude=20.1,
                altitude_msl=110.0,
                speed=20.0,
            ),
        ]

        cleaned_points, takeoff_time, landing_time = clean_flight_data(points)

        assert len(cleaned_points) == len(points)  # All points should be valid
        assert takeoff_time is not None
        assert landing_time is not None
