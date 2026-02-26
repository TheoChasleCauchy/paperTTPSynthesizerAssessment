import pandas as pd
import os
import librosa
import numpy as np
import soundfile as sf
import re

dataset_dir = "data/RWC/RWC-I/"
output_base_dir = "data/RWC/RWC-preprocessed/"

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
    Convert note name (e.g., 'A0', 'C#4') to MIDI number.
    """
    match = re.match(r"^([A-Ga-g][#b]?)(-?\d+)$", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")
    
    pitch_class, octave = match.groups()
    pitch_class = pitch_class.capitalize()
    octave = int(octave)
    
    midi_number = 12 * (octave + 1) + NOTE_OFFSETS[pitch_class]
    return midi_number

def semitone_range(note1, note2):
    """
    Returns the number of semitones between note1 and note2
    """
    midi1 = note_to_midi(note1)
    midi2 = note_to_midi(note2)
    return abs(midi2 - midi1) + 1 # +1 to include both endpoints

def split_into_notes(
    wav_path,
    base_name,
    sr=None,
    min_note_duration=0.010,
    top_db=80
):
    """
    Splits a monophonic scale recording into notes
    using silence detection instead of onset detection.

    Parameters:
        top_db: threshold (in dB) below reference to consider as silence.
                Lower = stricter silence detection.
    Returns:
        notes: list of (note_array, note_filename)
        sr: sampling rate
    """

    y, sr = librosa.load(wav_path, sr=sr)

    # --- Normalize audio (peak normalization) ---
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
    note_idx = 1

    for start, end in intervals:
        note = y[start:end]

        if len(note) / sr < min_note_duration:
            continue

        note_filename = f"{base_name}_{note_idx:02d}.wav"
        notes.append((note, note_filename))
        note_idx += 1

    return notes, sr

def safe_split_pitch_range(pr):
    try:
        parts = pr.split(">>")
        if len(parts) > 2:
            return pd.Series([np.nan, np.nan])
        if len(parts) == 1:
            return pd.Series([parts[0].strip(), parts[0].strip()])
        return pd.Series([parts[0].strip(), parts[1].strip()])
    except:
        # print('Error splitting pitch range:', parts)
        return pd.Series([np.nan, np.nan])

def safe_semitone_range(min_note, max_note):
    try:
        return semitone_range(min_note, max_note)
    except:
        return np.nan


################# MAIN #################

def preprocess_RWC():
    file_path = os.path.join(dataset_dir, "02_instruments_details_en.csv")

    df = pd.read_csv(file_path, sep=",")


    df["Instrument name"] = df["Instrument name"].ffill()

    # Work only on strings
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


    df = df[
        df["Playing style (articulation / method)"]
        .str.contains(r"normal|single|double", case=False, na=False)
    ]

    df = df[
        df["Instrument name"]
        .str.lower()
        .isin(ALLOWED_INSTRUMENTS)
    ]

    # Rename Dynamics columns
    df = df.rename(columns={
        'Dynamics (F: forte)': 'Dynamics (F)',
        'Dynamics (M: mezzo)': 'Dynamics (M)',
        'Dynamics (P: piano)': 'Dynamics (P)'
    })

    # Columns for the different types
    file_cols = ["File name (F)", "File name (M)", "File name (P)"]
    dynamics_cols = ["Dynamics (F)", "Dynamics (M)", "Dynamics (P)"]
    pitch_cols = ["Pitch range (F)", "Pitch range (M)", "Pitch range (P)"]
    length_cols = ["File length (F)", "File length (M)", "File length (P)"]

    # Melt file names
    df_files = df.melt(
        id_vars=[c for c in df.columns if c not in file_cols],
        value_vars=file_cols,
        var_name="File type",
        value_name="File name"
    )

    # Melt Dynamics
    df_dyn = df.melt(
        id_vars=[c for c in df.columns if c not in dynamics_cols],
        value_vars=dynamics_cols,
        var_name="File type",
        value_name="Dynamics"
    )

    # Melt Pitch range
    df_pitch = df.melt(
        id_vars=[c for c in df.columns if c not in pitch_cols],
        value_vars=pitch_cols,
        var_name="File type",
        value_name="Pitch range"
    )

    # Melt File length
    df_length = df.melt(
        id_vars=[c for c in df.columns if c not in length_cols],
        value_vars=length_cols,
        var_name="File type",
        value_name="File length"
    )

    # Extract the type F/M/P
    for df_melt in [df_files, df_dyn, df_pitch, df_length]:
        df_melt["File type"] = df_melt["File type"].str.extract(r"\((.)\)")

    # Merge all melted dataframes on common columns + File type
    merge_cols = [c for c in df.columns if c not in file_cols + dynamics_cols + pitch_cols + length_cols] + ["File type"]

    df = df_files.merge(df_dyn[merge_cols + ["Dynamics"]], on=merge_cols)
    df = df.merge(df_pitch[merge_cols + ["Pitch range"]], on=merge_cols)
    df = df.merge(df_length[merge_cols + ["File length"]], on=merge_cols)

    # Optional: remove rows where File name is NaN
    df = df.dropna(subset=["File name"])

    # Reorder columns to your desired format
    df = df[[
        "Inst. No.", "Variation No.", "Instrument name", "Instrument symbol",
        "Playing style (articulation / method)", "Playing style symbol",
        "Dynamics", "File type", "File name", "DVD Vol.", "Manufacturer",
        "Pitch range", "Number of JPEG files", "File length"
    ]]


    df[["Pitch min", "Pitch max"]] = df["Pitch range"].apply(safe_split_pitch_range)

    df = df.drop_duplicates(subset="File name", keep="first")

    # Apply row-wise
    df["Semitone range"] = df.apply(
        lambda row: safe_semitone_range(row["Pitch min"], row["Pitch max"]),
        axis=1
    )

    file_index = {}
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            file_index[f] = os.path.join(root, f)

    note_rows = []

    for idx, row in df.iterrows():
        filename = row["File name"]
        note_range = row["Semitone range"]
        instr_name = row["Instrument name"]    

        if pd.isna(filename):
            continue

        full_path = file_index.get(filename)

        if full_path is None:
            print(f"[WARNING] File not found: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_base_dir, instr_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            if note_range == 1:
                # Just copy/save the full file as a single note
                y, sr = librosa.load(full_path, sr=None)
                out_path = os.path.join(output_dir, f"{base_name}_01.wav")
                sf.write(out_path, y, sr)
                print(f"Processed {filename} (1 note, saved whole file)")
                continue  # skip to next row

            # For multi-note files, adjust threshold to match note_range
            threshold = 70
            notes, sr = split_into_notes(full_path, base_name, top_db=threshold)

            # Save the split notes
            for note_array, note_filename in notes:
                out_path = os.path.join(output_dir, note_filename)
                sf.write(out_path, note_array, sr)

                note_rows.append({
                    "Note file": note_filename,
                    "Instrument name": instr_name
                })

            print(f"Processed {filename} {output_dir} ({len(notes)} notes, {note_range} expected)")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    # Create a new DataFrame
    df_notes = pd.DataFrame(note_rows)

    # Count number of notes per instrument
    note_counts = df_notes["Instrument name"].value_counts()

    print(note_counts)