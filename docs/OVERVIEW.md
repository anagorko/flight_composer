# Flight Composer

The Flight Composer projects implements data processing and rendering pipeline to prepare clips for youtube videos from flight data recorded by GoPro cameras or logged with a Seeyou Navigator on a smartphone during the flight.

## Data Processing

The input data is organized according to flight uid from my logbook and short informative tag, e.g. `07_niskie_ladowanie`. Either `.mp4` or `.igc` file is present or both.

The data is processed as described in the [DATA_PROCESSING_PIPELINE](DATA_PROCESSING_PIPELINE.md) document. Here is a brief summary:

1. **Source File Discovery** — Locates GoPro `.mp4` and/or IGC `.igc` files for the given flight UID. GoPro is preferred when both are available.

2. **Telemetry Extraction** — Parses raw sensor data into Parquet files: GPS trajectory at ~18 Hz plus high-frequency IMU at ~200 Hz from GoPro, or 1 Hz B-record fixes from IGC.

3. **FlightTrack Construction** — Projects WGS 84 coordinates to a local East-North-Up frame, assigns data-quality weights, sanitizes glitches, validates the schema, classifies flight phases (debounced state machine + HMM/Viterbi), trims ground recordings, and corrects altitude drift.

4. **Spline Fitting** — Fits independent B-splines to the horizontal (quintic, k=5) and vertical (cubic, k=3) axes via `KinematicSpline` (FITPACK), producing a continuous, analytically differentiable `FlightTrajectory`.

5. **Orientation Computation** — Derives yaw, roll, and pitch from ground-track kinematics: heading from velocity direction, bank angle from centripetal acceleration, and pitch from flight-path angle plus aerodynamic angle of attack estimated using glider parameters.

6. **Trajectory DataFrame Assembly** — Samples the continuous trajectory at 60 Hz, computes felt G-load, interpolates flight phases via nearest-neighbor, and validates the result against a pandera schema.

7. **Export** — Writes the final trajectory DataFrame as both Parquet and CSV to the `ActorData` directory for consumption by the Unreal Engine visualization.

## Unreal Engine Actors

Actors read the exported `{flight_tag}_trajectory_df.csv` file and convert ENU coordinates to Unreal world space (East → +X, North → −Y, Up → +Z, scaled to centimetres). They are driven by a `CurrentAnimationTime` property keyable in Sequencer. See [ACTOR](ACTOR.md) for full details.

- **Glider Path Actor** (`AGliderPathActor`) — Builds a procedural ribbon mesh representing the flight path. The mesh is generated once at load time; a dynamic material uses the timestamp encoded in UVs to reveal, fade, and dim the trail behind the glider each frame.

- **Glider Playback Actor** (`AGliderPlaybackActor`) — Animates a static-mesh glider model along the trajectory. At load time it pre-computes a quaternion for each data point from the velocity vector and roll angle, and parses all telemetry columns (speed, accelerations, G-load, yaw, pitch, roll, flight phase). During playback it binary-searches for the surrounding frames and interpolates position (lerp), rotation (slerp), and all float telemetry fields (lerp); the phase string uses nearest-neighbour from the earlier sample. When `bExportOverlayData` is enabled, the actor exports a frame-accurate CSV during Movie Render Queue (MRQ) renders by querying `UMoviePipelineBlueprintLibrary::GetOverallOutputFrames()` for the true output frame index — which MRQ holds constant across TAA sub-samples, giving 1:1 correspondence with the output image sequence. The effective FPS is read from MRQ's pipeline config (custom frame rate or sequence display rate) to compute a `video_time_s` column. A stale-frame guard skips the engine frame on which `BeginPlay` fires, where Sequencer's first evaluation carries the old editor playhead time.
