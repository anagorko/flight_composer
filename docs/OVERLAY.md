# Flight Composer: Overlay Architecture

This document describes the Python overlay rendering pipeline, which takes raw telemetry data from Unreal Engine (via CSV) and generates transparent PNG sequences for use in DaVinci Resolve.

## 🔄 Workflow

The rendering pipeline is orchestrated by `src/flight_composer/scripts/render_overlays.py` and follows a strict four-step workflow:

1. **Ingestion & Validation:** The script dynamically locates the telemetry CSV using the sequence and actor names (`<SequenceName>_<ActorName>.csv`) in `config.DIR.SEQUENCE_DATA`. The data is loaded into a Pandas DataFrame and strictly validated using a Pandera schema to ensure all expected data types and columns are present.
2. **Preprocessing:** The validated DataFrame is passed through `src/flight_composer/overlays/preprocessing.py`. This step calculates derived metrics (like `speedup_raw`) and injects them as new columns into the DataFrame *before* any widgets see the data.
3. **Widget Factory:** The script looks up the sequence's configuration from `config.sequences[sequence_name]` (a `SequenceConfig` instance, see [Configuration](#configuration) below). For every widget defined in the config, it instantiates the corresponding Python class (inheriting from the `Overlay` base class).
4. **Render Loop:** The script iterates through the DataFrame row by row. For each row (representing exactly one video frame), it creates a blank RGBA canvas for each active widget, calls the widget's `render_frame()` method, and saves the output as `{widget_name}_{frame_id}.png` to `config.DIR.OVERLAY / sequence_name`.

## 🏗️ Core Concepts

* **The Base `Overlay` Class** (`src/flight_composer/overlay/overlay.py`): All UI elements inherit from this abstract base class. It provides standard utilities for coordinate parsing (handling both pixels and percentages), font loading with fallbacks, and a standardized method for drawing legible, outlined text against complex video backgrounds.
* **1 Row = 1 Frame:** Thanks to C++ MRQ integration, the Python system blindly trusts the CSV. There is no complex time-warping or sub-sample deduplication in Python; if a row exists, it gets rendered as a frame.

## Configuration

All sequence overlay configurations live as YAML files in `config/sequences/` (e.g. `Seq_EPBC_Overhead_Intro.yaml`). At startup, `load_config()` in `src/flight_composer/config.py` scans this directory and loads every `.yaml` file into `config.sequences` — a `dict[str, SequenceConfig]` keyed by the file stem.

The `SequenceConfig` and `SequenceGlobalConfig` Pydantic models in `config.py` define the expected structure. See those classes for the authoritative schema; the key ideas are summarised below.

### Global Block

The `global` block defines the base canvas and fallback font size:

```yaml
global:
  resolution: [1920, 1080]
  font_size: 24
```

Font *paths* are resolved from `config.DIR.FONT_PATH`, not from the YAML.

### Widgets List

The `widgets` block is a list of widget definitions. Each entry requires at minimum a `type` (which maps to a Python class) and a `name` (which dictates the output file prefix). All other keys are type-specific and passed to the widget constructor as a config dict.

**Coordinate rules (`x` and `y`):**
* **Integers:** Absolute pixels from the top-left (e.g., `x: 50` means 50 pixels from the left).
* **Strings with `%`:** Percentage of the global resolution (e.g., `x: "50%"` on a 1920×1080 canvas resolves to 960).

**Example widget definition:**

```yaml
widgets:
  - type: "altitude"
    name: "alt_display"
    x: "10%"
    y: 950
    title_text: "Alt [m]"
    title_color: "#AAAAAA"
    title_font_size: 24
    value_color: "white"
    value_font_size: 72
```

A complete example config is in `config/sequences/Seq_EPBC_Overhead_Intro.yaml`.

## 🧰 Available Widgets

Widget implementations live in `src/flight_composer/overlay/`. Each class inherits from `Overlay`.

* **`overview`** (`OverviewOverlay`): A master debug widget that renders every single column (including preprocessed derived columns) as a vertical list. Always rendered automatically.
* **`speedup`** (`SpeedupOverlay`): Calculates the ratio of real flight time to video time to display the current animation playback speed (e.g., "8x >>" or "0.5x <<"). Can be rounded to nearest powers of 2 via the `round` config key.
* **`altitude`** (`AltitudeOverlay`): A two-line display showing a customizable title and the glider's clamped altitude (`z_m`). Uses distinct configurable font sizes and colors for the title and value.
* **`speed`** (`SpeedOverlay`): A two-line display showing ground speed. Supports configurable units (`km/h` or `m/s`) via the `unit` config key; conversion is handled automatically.