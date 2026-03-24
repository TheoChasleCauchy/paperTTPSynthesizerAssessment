import pandas as pd
import os
import librosa
import numpy as np
import soundfile as sf  # Library for reading and writing sound files
import re  # Regular expressions for string manipulation
from tqdm import tqdm  # Progress bar for iterative tasks

# Directory containing the RWC dataset
dataset_dir = "data/RWC/RWC-I/"
# Base directory for saving preprocessed audio files
output_base_dir = "data/RWC/RWC-preprocessed/"

# Dictionary mapping note names to their MIDI offsets
NOTE_OFFSETS = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11
}

# List of allowed instruments for preprocessing
ALLOWED_INSTRUMENTS = [
    "piccolo",
    "flute",
    "oboe",
    "english horn",
    "clarinet",
    "soprano sax",
    "alto sax",
    "tenor sax",
    "baritone sax",
    "bassoon",
    "horn",
    "trumpet",
    "trombone",
    "tuba",
    "timpani",
    "bass drum",
    "snare drum",
    "glockenspiel",
    "xylophone",
    "vibraphone",
    "marimba",
    "crash cymbal",
    "triangle",
    "wood block",
    "pianoforte",
    "harpsichord",
    "harp",
    "violin",
    "viola",
    "cello",
    "contrabass",
]

def note_to_midi(note):
    """
    Convert a note name (e.g., 'A0', 'C#4') to its corresponding MIDI number.

    Args:
        note (str): Note name in the format 'NoteOctave' (e.g., 'C4', 'F#3').

    Returns:
        int: MIDI number corresponding to the note.

    Raises:
        ValueError: If the note format is invalid.
    """
    # Use regex to extract pitch class and octave from the note string
    match = re.match(r"^([A-Ga-g][#b]?)(-?\d+)$", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")

    pitch_class, octave = match.groups()
    pitch_class = pitch_class.capitalize()  # Ensure pitch class is capitalized
    octave = int(octave)  # Convert octave to integer

    # Calculate MIDI number using the formula: 12 * (octave + 1) + NOTE_OFFSET
    midi_number = 12 * (octave + 1) + NOTE_OFFSETS[pitch_class]
    return midi_number

def semitone_range(note1, note2):
    """
    Calculate the number of semitones between two notes, inclusive.

    Args:
        note1 (str): First note in the format 'NoteOctave'.
        note2 (str): Second note in the format 'NoteOctave'.

    Returns:
        int: Number of semitones between the two notes.
    """
    # Convert notes to MIDI numbers and calculate the absolute difference
    midi1 = note_to_midi(note1)
    midi2 = note_to_midi(note2)
    return abs(midi2 - midi1) + 1  # +1 to include both endpoints

def split_into_notes(
    wav_path,
    base_name,
    sr=None,
    min_note_duration=0.010,
    top_db=80
):
    """
    Split a monophonic scale recording into individual notes using silence detection.

    Args:
        wav_path (str): Path to the WAV file.
        base_name (str): Base name for output note files.
        sr (int, optional): Sample rate. If None, use the file's native sample rate.
        min_note_duration (float, optional): Minimum duration (in seconds) for a note to be considered valid. Defaults to 0.010.
        top_db (int, optional): Threshold (in dB) below reference to consider as silence. Defaults to 80.

    Returns:
        tuple: A tuple containing:
            - notes (list): List of tuples, each containing a note array and its filename.
            - sr (int): Sample rate of the audio.
    """
    # Load audio file
    y, sr = librosa.load(wav_path, sr=sr)

    # Normalize audio to peak amplitude
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Detect non-silent intervals
    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=int(sr/2),
        hop_length=int(sr/4)
    )

    notes = []
    note_idx = 1  # Index for naming output note files

    # Iterate over detected intervals and extract notes
    for start, end in intervals:
        note = y[start:end]

        # Skip intervals shorter than the minimum note duration
        if len(note) / sr < min_note_duration:
            continue

        # Generate output filename for the note
        note_filename = f"{base_name}_{note_idx:02d}.wav"
        notes.append((note, note_filename))
        note_idx += 1

    return notes, sr

def safe_split_pitch_range(pr):
    """
    Safely split a pitch range string into minimum and maximum notes.

    Args:
        pr (str): Pitch range string in the format "min_note>>max_note".

    Returns:
        pd.Series: A pandas Series containing the minimum and maximum notes.
                  Returns NaN for both if the format is invalid.
    """
    try:
        parts = pr.split(">>")
        if len(parts) > 2:
            return pd.Series([np.nan, np.nan])
        if len(parts) == 1:
            return pd.Series([parts[0].strip(), parts[0].strip()])
        return pd.Series([parts[0].strip(), parts[1].strip()])
    except:
        return pd.Series([np.nan, np.nan])

def safe_semitone_range(min_note, max_note):
    """
    Safely calculate the semitone range between two notes.

    Args:
        min_note (str): Minimum note in the range.
        max_note (str): Maximum note in the range.

    Returns:
        float: Number of semitones between the notes. Returns NaN if calculation fails.
    """
    try:
        return semitone_range(min_note, max_note)
    except:
        return np.nan

################# MAIN #################

def preprocess_RWC():
    """
    Preprocess the RWC (Real World Computing) Musical Instrument Sound Database.
    This function performs the following operations:
    1. Loads the RWC instruments details CSV file.
    2. Cleans and standardizes instrument names.
    3. Filters the dataset to include only relevant instruments and playing styles.
    4. Reshapes the data by melting separate columns for dynamics, file names, pitch ranges, and file lengths.
    5. Merges the melted dataframes to create a normalized structure.
    6. Extracts pitch range information and calculates semitone ranges.
    7. Deduplicates entries based on file names.
    8. Builds an index of all audio files in the dataset directory.
    9. Splits multi-note audio files into individual notes using silence detection.
    10. Saves split notes to the output directory organized by instrument.
    11. Creates a summary DataFrame with note file information and associated instrument names.
    12. Prints statistics on the number of notes per instrument.

    Returns:
        None: Outputs audio files to disk and prints statistics.
    """
    print("[INFO] Preprocessing RWC dataset.")

    # Load the instruments details CSV file
    file_path = os.path.join(dataset_dir, "02_instruments_details_en.csv")
    df = pd.read_csv(file_path, sep=",")

    # Forward-fill missing instrument names
    df["Instrument name"] = df["Instrument name"].ffill()

    # Ensure instrument names are strings
    df["Instrument name"] = df["Instrument name"].astype(str)

    # If there is a "/", keep only the part after it
    df["Instrument name"] = df["Instrument name"].str.split("/", n=1).str[-1].str.strip()

    # Remove anything in parentheses
    df["Instrument name"] = df["Instrument name"].str.replace(
        r"\([^)]*\)", "", regex=True
    )

    # Remove everything from the first digit onward (digit included)
    df["Instrument name"] = df["Instrument name"].str.replace(
        r"\d.*", "", regex=True
    ).str.strip()

    # Filter for relevant playing styles
    df = df[
        df["Playing style (articulation / method)"]
        .str.contains(r"normal|single|double", case=False, na=False)
    ]

    # Filter for allowed instruments
    df = df[
        df["Instrument name"]
        .str.lower()
        .isin([instr.lower() for instr in ALLOWED_INSTRUMENTS])
    ]

    # Rename dynamics columns for consistency
    df = df.rename(columns={
        'Dynamics (F: forte)': 'Dynamics (F)',
        'Dynamics (M: mezzo)': 'Dynamics (M)',
        'Dynamics (P: piano)': 'Dynamics (P)'
    })

    # Define columns for different data types
    file_cols = ["File name (F)", "File name (M)", "File name (P)"]
    dynamics_cols = ["Dynamics (F)", "Dynamics (M)", "Dynamics (P)"]
    pitch_cols = ["Pitch range (F)", "Pitch range (M)", "Pitch range (P)"]
    length_cols = ["File length (F)", "File length (M)", "File length (P)"]

    # Melt file names into a single column
    df_files = df.melt(
        id_vars=[c for c in df.columns if c not in file_cols],
        value_vars=file_cols,
        var_name="File type",
        value_name="File name"
    )

    # Melt dynamics into a single column
    df_dyn = df.melt(
        id_vars=[c for c in df.columns if c not in dynamics_cols],
        value_vars=dynamics_cols,
        var_name="File type",
        value_name="Dynamics"
    )

    # Melt pitch ranges into a single column
    df_pitch = df.melt(
        id_vars=[c for c in df.columns if c not in pitch_cols],
        value_vars=pitch_cols,
        var_name="File type",
        value_name="Pitch range"
    )

    # Melt file lengths into a single column
    df_length = df.melt(
        id_vars=[c for c in df.columns if c not in length_cols],
        value_vars=length_cols,
        var_name="File type",
        value_name="File length"
    )

    # Extract the dynamics type (F/M/P) from the "File type" column
    for df_melt in [df_files, df_dyn, df_pitch, df_length]:
        df_melt["File type"] = df_melt["File type"].str.extract(r"\((.)\)")

    # Merge all melted dataframes on common columns + File type
    merge_cols = [c for c in df.columns if c not in file_cols + dynamics_cols + pitch_cols + length_cols] + ["File type"]

    df = df_files.merge(df_dyn[merge_cols + ["Dynamics"]], on=merge_cols)
    df = df.merge(df_pitch[merge_cols + ["Pitch range"]], on=merge_cols)
    df = df.merge(df_length[merge_cols + ["File length"]], on=merge_cols)

    # Remove rows where File name is NaN
    df = df.dropna(subset=["File name"])

    # Reorder columns for clarity
    df = df[[
        "Inst. No.", "Variation No.", "Instrument name", "Instrument symbol",
        "Playing style (articulation / method)", "Playing style symbol",
        "Dynamics", "File type", "File name", "DVD Vol.", "Manufacturer",
        "Pitch range", "Number of JPEG files", "File length"
    ]]

    # Split pitch range into minimum and maximum notes
    df[["Pitch min", "Pitch max"]] = df["Pitch range"].apply(safe_split_pitch_range)

    # Remove duplicate entries based on file name
    df = df.drop_duplicates(subset="File name", keep="first")

    # Calculate semitone range for each entry
    df["Semitone range"] = df.apply(
        lambda row: safe_semitone_range(row["Pitch min"], row["Pitch max"]),
        axis=1
    )

    # Build an index of all audio files in the dataset directory
    file_index = {}
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            file_index[f] = os.path.join(root, f)

    note_rows = []  # List to store information about split notes

    # Iterate over each row in the DataFrame and process audio files
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting scales into individual notes"):
        filename = row["File name"]
        note_range = row["Semitone range"]
        instr_name = row["Instrument name"]

        # Skip rows with missing filenames
        if pd.isna(filename):
            continue

        # Get the full path to the audio file
        full_path = file_index.get(filename)

        if full_path is None:
            print(f"[WARNING] File not found: {filename}")
            continue

        # Create output directory for the instrument
        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_base_dir, instr_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # If the note range is 1, save the entire file as a single note
            if note_range == 1:
                y, sr = librosa.load(full_path, sr=None)
                out_path = os.path.join(output_dir, f"{base_name}_01.wav")
                sf.write(out_path, y, sr)
                print(f"Processed {filename} (1 note, saved whole file)")
                continue  # Skip to the next row

            # For multi-note files, split into individual notes
            threshold = 70
            notes, sr = split_into_notes(full_path, base_name, top_db=threshold)
            note_lengths = [len(note_array)/sr for note_array, _ in notes]

            # Adjust threshold if notes are too long
            while any(length > 15 for length in note_lengths):
                threshold -= 5
                notes, sr = split_into_notes(full_path, base_name, top_db=threshold)
                note_lengths = [len(note_array)/sr for note_array, _ in notes]

            # Save the split notes to disk
            for note_array, note_filename in notes:
                if len(note_array)/sr > 15:
                    print(filename)
                    print(note_filename)
                    print(len(note_array)/sr)

                out_path = os.path.join(output_dir, note_filename)
                sf.write(out_path, note_array, sr)

                # Record information about the note
                note_rows.append({
                    "Note file": note_filename,
                    "Instrument name": instr_name
                })

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    # Create a DataFrame from the note information
    df_notes = pd.DataFrame(note_rows)

    # Count and print the number of notes per instrument
    note_counts = df_notes["Instrument name"].value_counts()
    print(note_counts)
