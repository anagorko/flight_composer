# Flight Composer: Overlay Architecture

This document describes the Python overlay rendering pipeline, which takes raw telemetry data from Unreal Engine (via CSV) and generates transparent PNG sequences for use in DaVinci Resolve.

## 🔄 Workflow

The rendering pipeline is orchestrated by `src/flight_composer/scripts/render_overlays.py` and follows a strict four-step workflow:

1. **Ingestion & Validation:** The script dynamically locates the telemetry CSV using the sequence and actor names (`<SequenceName>_<ActorName>.csv`). The data is loaded into a Pandas DataFrame and strictly validated using a Pandera schema to ensure all expected data types and columns are present.
2. **Preprocessing:** The validated DataFrame is passed through `src/flight_composer/overlays/preprocessing.py`. This step calculates derived metrics (like `speedup_raw`) and injects them as new columns into the DataFrame *before* any widgets see the data.
3. **Widget Factory:** The script reads the sequence's YAML configuration file. For every widget defined in the YAML, it instantiates the corresponding Python class (inheriting from the `Overlay` base class).
4. **Render Loop:** The script iterates through the DataFrame row by row. For each row (representing exactly one video frame), it creates a blank RGBA canvas for each active widget, calls the widget's `render_frame()` method, and saves the output as `{widget_name}_{frame_id}.png`.

## 🏗️ Core Concepts

* **The Base `Overlay` Class:** All UI elements inherit from this abstract base class. It provides standard utilities for coordinate parsing (handling both pixels and percentages), font loading with fallbacks, and a standardized method for drawing legible, outlined text against complex video backgrounds.
* **1 Row = 1 Frame:** Thanks to C++ MRQ integration, the Python system blindly trusts the CSV. There is no complex time-warping or sub-sample deduplication in Python; if a row exists, it gets rendered as a frame.

## 📝 YAML Configuration Rules

The layout and styling of the overlays are entirely controlled by a sequence-specific YAML file (e.g., `Seq_EPBC_Overhead_Intro.yaml`). 

### Global Block
The `global` block defines the base canvas and fallbacks:

```yaml
global:
  resolution: [1920, 1080] # The output size of the PNGs
  font_size: 24            # The fallback font size if a widget doesn't specify one
```

*(Note: Font paths are managed globally via the Python `config` object, not in the YAML).*

### Widgets List
The `widgets` block contains a list of UI elements to render. Each widget requires a `type` (which maps to a Python class) and a `name` (which dictates the output file prefix).

**Coordinate Rules (`x` and `y`):**
* **Integers:** Treated as absolute pixels from the top-left (e.g., `x: 50` means 50 pixels from the left).
* **Strings with `%`:** Treated as a percentage of the global resolution (e.g., `x: "50%"` on a 1920x1080 canvas resolves to 960).

**Example Widget Definition:**
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

## 🧰 Available Widgets

* **`overview`**: A master debug widget that renders every single column (including preprocessed derived columns) as a vertical list. Always rendered automatically.
* **`speedup`**: Calculates the ratio of real flight time to video time to display the current animation playback speed (e.g., "8x >>" or "0.5x <<"). Can be rounded to nearest powers of 2.
* **`altitude`**: A two-line display showing a customizable title and the glider's clamped altitude (`z_m`). Uses distinct configurable font sizes and colors for the title and value.
