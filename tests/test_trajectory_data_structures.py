"""
Unit tests for trajectory data structures.

Tests the core data structures used in flight trajectory processing,
including TrajectoryPoint, NormalizedFlightData, and related classes.
"""

from datetime import datetime

import numpy as np
import pytest

from flight_composer.trajectory.data_structures import (
    CoordinateOrigin,
    FlightMetadata,
    FlightSegment,
    NormalizedFlightData,
    RawFlightPoint,
    TrajectoryPoint,
    TurbulenceDelta,
)


class TestTrajectoryPoint:
    """Test TrajectoryPoint data structure."""

    def test_trajectory_point_creation(self):
        """Test basic TrajectoryPoint creation."""
        point = TrajectoryPoint(
            timestamp=1000.0,
            x=100.5,
            y=200.5,
            z=50.0,
            source="TEST",
            quality=0.95,
        )

        assert point.timestamp == 1000.0
        assert point.x == 100.5
        assert point.y == 200.5
        assert point.z == 50.0
        assert point.source == "TEST"
        assert point.quality == 0.95

    def test_trajectory_point_defaults(self):
        """Test TrajectoryPoint with default values."""
        point = TrajectoryPoint(timestamp=1000.0, x=0.0, y=0.0, z=0.0)

        assert point.source == "UNKNOWN"
        assert point.quality == 1.0


class TestTurbulenceDelta:
    """Test TurbulenceDelta data structure."""

    def test_turbulence_delta_creation(self):
        """Test TurbulenceDelta creation."""
        delta = TurbulenceDelta(
            timestamp=1000.0,
            rot_x=0.01,
            rot_y=-0.02,
            z_jitter=0.5,
            source="GPX_IMU",
        )

        assert delta.timestamp == 1000.0
        assert delta.rot_x == 0.01
        assert delta.rot_y == -0.02
        assert delta.z_jitter == 0.5
        assert delta.source == "GPX_IMU"

    def test_turbulence_delta_defaults(self):
        """Test TurbulenceDelta with defaults."""
        delta = TurbulenceDelta(timestamp=1000.0, rot_x=0.0, rot_y=0.0, z_jitter=0.0)

        assert delta.source == "UNKNOWN"


class TestCoordinateOrigin:
    """Test CoordinateOrigin data structure."""

    def test_coordinate_origin_creation(self):
        """Test CoordinateOrigin creation."""
        origin = CoordinateOrigin(
            latitude=52.123456,
            longitude=20.987654,
            altitude_msl=123.5,
            projection="aeqd",
        )

        assert origin.latitude == 52.123456
        assert origin.longitude == 20.987654
        assert origin.altitude_msl == 123.5
        assert origin.projection == "aeqd"

    def test_coordinate_origin_defaults(self):
        """Test CoordinateOrigin with defaults."""
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)

        assert origin.projection == "aeqd"


class TestFlightMetadata:
    """Test FlightMetadata data structure."""

    def test_flight_metadata_creation(self):
        """Test FlightMetadata creation."""
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)

        metadata = FlightMetadata(
            origin=origin,
            glider_polar={"type": "Test Glider"},
            wind_vector=(5.0, -2.0, 0.1),
            terrain_offset=10.0,
            takeoff_time=1000.0,
            landing_time=2000.0,
            flight_number="07",
        )

        assert metadata.origin == origin
        assert metadata.glider_polar == {"type": "Test Glider"}
        assert metadata.wind_vector == (5.0, -2.0, 0.1)
        assert metadata.terrain_offset == 10.0
        assert metadata.takeoff_time == 1000.0
        assert metadata.landing_time == 2000.0
        assert metadata.flight_number == "07"

    def test_flight_metadata_defaults(self):
        """Test FlightMetadata with defaults."""
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)
        metadata = FlightMetadata(origin=origin)

        assert metadata.glider_polar is None
        assert metadata.wind_vector is None
        assert metadata.terrain_offset == 0.0
        assert metadata.takeoff_time is None
        assert metadata.landing_time is None
        assert metadata.flight_number is None


class TestNormalizedFlightData:
    """Test NormalizedFlightData data structure."""

    def create_sample_data(self) -> NormalizedFlightData:
        """Create sample NormalizedFlightData for testing."""
        # Create sample trajectory points
        trajectory_points = [
            TrajectoryPoint(timestamp=1000.0, x=0.0, y=0.0, z=0.0, source="TEST"),
            TrajectoryPoint(timestamp=1010.0, x=100.0, y=50.0, z=10.0, source="TEST"),
            TrajectoryPoint(timestamp=1020.0, x=200.0, y=100.0, z=20.0, source="TEST"),
        ]

        # Create sample turbulence data
        turbulence_deltas = [
            TurbulenceDelta(
                timestamp=1005.0, rot_x=0.01, rot_y=-0.02, z_jitter=0.5, source="TEST"
            ),
            TurbulenceDelta(
                timestamp=1015.0, rot_x=-0.01, rot_y=0.01, z_jitter=-0.3, source="TEST"
            ),
        ]

        # Create metadata
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)
        metadata = FlightMetadata(origin=origin, flight_number="TEST")

        return NormalizedFlightData(
            trajectory_points=trajectory_points,
            turbulence_deltas=turbulence_deltas,
            metadata=metadata,
        )

    def test_normalized_flight_data_creation(self):
        """Test NormalizedFlightData creation."""
        data = self.create_sample_data()

        assert len(data.trajectory_points) == 3
        assert len(data.turbulence_deltas) == 2
        assert data.metadata.flight_number == "TEST"
        assert data.data_gaps == []
        assert data.processing_notes == []

    def test_normalized_flight_data_properties(self):
        """Test NormalizedFlightData computed properties."""
        data = self.create_sample_data()

        # Test duration
        assert data.duration == 20.0  # 1020 - 1000

        # Test start/end times
        assert data.start_time == 1000.0
        assert data.end_time == 1020.0

    def test_normalized_flight_data_empty(self):
        """Test NormalizedFlightData with empty trajectory."""
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)
        metadata = FlightMetadata(origin=origin)

        data = NormalizedFlightData(
            trajectory_points=[],
            turbulence_deltas=[],
            metadata=metadata,
        )

        assert data.duration == 0.0
        assert data.start_time is None
        assert data.end_time is None

    def test_get_trajectory_arrays(self):
        """Test trajectory data extraction as numpy arrays."""
        data = self.create_sample_data()

        timestamps, x_coords, y_coords, z_coords = data.get_trajectory_arrays()

        # Check shapes
        assert len(timestamps) == 3
        assert len(x_coords) == 3
        assert len(y_coords) == 3
        assert len(z_coords) == 3

        # Check values
        np.testing.assert_array_equal(timestamps, [1000.0, 1010.0, 1020.0])
        np.testing.assert_array_equal(x_coords, [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(y_coords, [0.0, 50.0, 100.0])
        np.testing.assert_array_equal(z_coords, [0.0, 10.0, 20.0])

    def test_get_trajectory_arrays_empty(self):
        """Test trajectory arrays with empty data."""
        origin = CoordinateOrigin(latitude=52.0, longitude=20.0, altitude_msl=100.0)
        metadata = FlightMetadata(origin=origin)

        data = NormalizedFlightData(
            trajectory_points=[],
            turbulence_deltas=[],
            metadata=metadata,
        )

        timestamps, x_coords, y_coords, z_coords = data.get_trajectory_arrays()

        assert len(timestamps) == 0
        assert len(x_coords) == 0
        assert len(y_coords) == 0
        assert len(z_coords) == 0

    def test_add_processing_note(self):
        """Test adding processing notes."""
        data = self.create_sample_data()

        data.add_processing_note("Test note 1")
        data.add_processing_note("Test note 2")

        assert len(data.processing_notes) == 2
        assert "Test note 1" in data.processing_notes
        assert "Test note 2" in data.processing_notes

    def test_add_data_gap(self):
        """Test adding data gaps."""
        data = self.create_sample_data()

        data.add_data_gap(1005.0, 1008.0)
        data.add_data_gap(1015.0, 1018.0)

        assert len(data.data_gaps) == 2
        assert (1005.0, 1008.0) in data.data_gaps
        assert (1015.0, 1018.0) in data.data_gaps


class TestRawFlightPoint:
    """Test RawFlightPoint data structure."""

    def test_raw_flight_point_creation(self):
        """Test RawFlightPoint creation."""
        point = RawFlightPoint(
            timestamp=1000.0,
            latitude=52.123456,
            longitude=20.987654,
            altitude_msl=150.5,
            source="IGC",
            hdop=2.5,
            speed=25.0,
            heading=180.5,
        )

        assert point.timestamp == 1000.0
        assert point.latitude == 52.123456
        assert point.longitude == 20.987654
        assert point.altitude_msl == 150.5
        assert point.source == "IGC"
        assert point.hdop == 2.5
        assert point.speed == 25.0
        assert point.heading == 180.5

    def test_raw_flight_point_defaults(self):
        """Test RawFlightPoint with defaults."""
        point = RawFlightPoint(
            timestamp=1000.0,
            latitude=52.0,
            longitude=20.0,
            altitude_msl=100.0,
        )

        assert point.source == "UNKNOWN"
        assert point.hdop is None
        assert point.speed is None
        assert point.heading is None


class TestFlightSegment:
    """Test FlightSegment data structure."""

    def test_flight_segment_creation(self):
        """Test FlightSegment creation."""
        points = [
            TrajectoryPoint(timestamp=1000.0, x=0.0, y=0.0, z=0.0, source="TEST"),
            TrajectoryPoint(timestamp=1010.0, x=100.0, y=50.0, z=10.0, source="TEST"),
        ]

        segment = FlightSegment(
            start_time=1000.0,
            end_time=1010.0,
            trajectory_points=points,
            segment_index=0,
        )

        assert segment.start_time == 1000.0
        assert segment.end_time == 1010.0
        assert len(segment.trajectory_points) == 2
        assert segment.segment_index == 0

    def test_flight_segment_properties(self):
        """Test FlightSegment computed properties."""
        points = [
            TrajectoryPoint(timestamp=1000.0, x=0.0, y=0.0, z=0.0, source="TEST"),
            TrajectoryPoint(timestamp=1010.0, x=100.0, y=50.0, z=10.0, source="TEST"),
            TrajectoryPoint(timestamp=1020.0, x=200.0, y=100.0, z=20.0, source="TEST"),
        ]

        segment = FlightSegment(
            start_time=1000.0,
            end_time=1020.0,
            trajectory_points=points,
            segment_index=1,
        )

        assert segment.duration == 20.0
        assert segment.point_count == 3

    def test_flight_segment_empty(self):
        """Test FlightSegment with no points."""
        segment = FlightSegment(
            start_time=1000.0,
            end_time=1010.0,
            trajectory_points=[],
            segment_index=0,
        )

        assert segment.duration == 10.0
        assert segment.point_count == 0
