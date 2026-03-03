import pandas as pd


def format_srt_time(seconds: float) -> str:
    """
    Converts a time in seconds to the SRT time format: HH:MM:SS,mmm
    """
    if pd.isna(seconds):
        return "00:00:00,000"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    # Extract milliseconds and round to avoid floating point weirdness
    millis = int(round((seconds - int(seconds)) * 1000))

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_phase_srt(
    df: pd.DataFrame,
    output_filepath: str,
    time_col: str = "t",
    phase_col: str = "phase_hmm",
):
    """
    Generates an SRT subtitle file from a DataFrame annotated with flight phases.
    """
    # 1. Create a grouping key that increments every time the phase changes.
    # This correctly separates recurring phases (e.g., CRUISING -> THERMALLING -> CRUISING)
    df_clean = df.dropna(subset=[time_col, phase_col]).copy()
    df_clean["phase_block"] = (
        df_clean[phase_col] != df_clean[phase_col].shift(1)
    ).cumsum()

    # 2. Aggregate the data to find the start and end time of each contiguous block
    blocks = (
        df_clean.groupby("phase_block")
        .agg(
            start_time=(time_col, "first"),
            end_time=(time_col, "last"),
            phase=(phase_col, "first"),
        )
        .reset_index()
    )

    # 3. Write out to the .srt file format
    with open(output_filepath, "w", encoding="utf-8") as f:
        for i, row in blocks.iterrows():
            start_str = format_srt_time(row["start_time"])
            end_str = format_srt_time(row["end_time"])

            # SRT Format requires:
            # Block Number
            # Start Time --> End Time
            # Subtitle Text
            # Blank Line
            f.write(f"{i + 1}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{row['phase']} until {end_str}\n\n")

    print(
        f"Successfully generated {len(blocks)} subtitle blocks in '{output_filepath}'."
    )


# ==========================================
# Example Usage:
# ==========================================
# Assuming 'df' is the dataframe returned by your assign_flight_phases_hmm function
# generate_phase_srt(df, "flight_phases_overlay.srt", time_col="t", phase_col="phase_hmm")
