import pandas as pd

def bemol_to_sharp(note_str: str) -> str:
    """Convert a note string with a flat (b) to its sharp equivalent."""
    if "Bb" in note_str:
        return note_str.replace("Bb", "A#")
    elif "Eb" in note_str:
        return note_str.replace("Eb", "D#")
    elif "Ab" in note_str:
        return note_str.replace("Ab", "G#")
    elif "Db" in note_str:
        return note_str.replace("Db", "C#")
    elif "Gb" in note_str:
        return note_str.replace("Gb", "F#")
    else:
        return note_str

def note_to_midi(note_str: str) -> int:
    """
    Convert a musical note string (e.g., "C#4", "F5") to its corresponding MIDI number.

    Args:
        note_str (str): The note and octave as a string, e.g., "C#4", "F5".

    Returns:
        int: The MIDI number corresponding to the note and octave.

    Raises:
        ValueError: If the note string is invalid or the MIDI number is out of bounds (0-127).
    """
    # List of all notes in an octave, including sharps.
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Split the note string into the note and octave parts.
    # Handle both single-letter notes (e.g., "C4") and notes with sharps/flats (e.g., "C#4").
    note_str = bemol_to_sharp(note_str)
    note_part = note_str[:-1].upper()  # Everything except the last character is the note.
    octave_part = note_str[-1]          # The last character is the octave.

    # Validate the octave is a digit.
    if not octave_part.isdigit():
        raise ValueError(f"Invalid octave: {octave_part}, note: {note_str}")

    octave = int(octave_part)

    # Find the index of the note in the list.
    try:
        note_index = notes.index(note_part)
    except ValueError:
        raise ValueError(f"Invalid note: {note_part}")

    # Calculate the MIDI number using "A4" = 69 as the reference.
    midi_number = 69 + (octave - 4) * 12 + (note_index - 9)

    # Check if the MIDI number is within valid bounds (0-127).
    if midi_number < 0 or midi_number > 127:
        raise ValueError(f"MIDI number {midi_number} is out of bounds (0-127)")

    return midi_number

def midi_to_note(midi_number: int) -> str:
    """
    Convert a MIDI number to its corresponding musical note and octave.

    Args:
        midi_number (int): The MIDI number to convert (0-127).

    Returns:
        str: The note and octave as a string, e.g., "C4", "F#5".

    Raises:
        ValueError: If the MIDI number is out of bounds (0-127).
    """
    # List of all notes in an octave, including sharps.
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Check if the MIDI number is within valid bounds (0-127).
    if midi_number < 0 or midi_number > 127:
        raise ValueError(f"MIDI number {midi_number} is out of bounds (0-127)")

    # Calculate the octave and note index.
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12

    # Get the note name from the list.
    note_name = notes[note_index]

    # Return the note and octave as a string.
    return f"{note_name}{octave}"

def get_midi_range_from_sample(sample_name: str) -> tuple[int, int]:
    """Return the MIDI range (min, max) for a given sample name."""
    # Get the metadata csv file
    instrument_metadata = pd.read_csv("resources/02_instruments_details_en.csv")

    # Get the last char before ".wav" in sample_name:
    sample_name = sample_name.replace(".wav", "")
    intensity_char = sample_name[-1]

    # Find the row for the given sample name
    match intensity_char:
        case "F":
            column_name = 'File name (F)'
        case "M":
            column_name = 'File name (M)'
        case "S":
            column_name = 'File name (S)'
        case _:
            raise ValueError(f"Unknown intensity character '{intensity_char}' in sample name '{sample_name}'")

    sample_name = sample_name + ".WAV"
    row = instrument_metadata[instrument_metadata[column_name] == sample_name]

    match intensity_char:
        case "F":
            column_name = 'Pitch range (F)'
        case "M":
            column_name = 'Pitch range (M)'
        case "S":
            column_name = 'Pitch range (S)'
        case _:
            raise ValueError(f"Unknown intensity character '{intensity_char}' in sample name '{sample_name}'")

    pitch_range = row[column_name].iloc[0]
    min_pitch = pitch_range[:2]
    max_pitch = pitch_range[-2:]

    print(f"Sample: {sample_name}, Min Pitch: {min_pitch}, Max Pitch: {max_pitch}")
    
    # Return the min and max MIDI values for the sample
    return (note_to_midi(min_pitch), note_to_midi(max_pitch))

def get_midi_range_from_instrument(instrument: str) -> tuple[int, int]:
    """Return the MIDI range (min, max) for a given sample name."""

    # Instruments which have no ranges
    if instrument in ["BASS DRUM", "SNARE DRUM", "CRASH CYMBAL", "TRIANGLE", "WOOD BLOCK"]:
        return (64,64)

    # Get the metadata csv file
    instrument_metadata = pd.read_csv("resources/02_instruments_details_en.csv")

    min_midis = []
    max_midis = []

    for instrument_name in instruments_names[instrument]:
        mask = instrument_metadata['Instrument name'] == instrument_name
        # Drop N/A values
        mask = mask & instrument_metadata['Instrument name'].notna()
        rows = instrument_metadata[mask]
        for intensity_char in ["F", "M", "P"]:
            for index, row in rows.iterrows():
                match intensity_char:
                    case "F":
                        column_name = 'Pitch range (F)'
                    case "M":
                        column_name = 'Pitch range (M)'
                    case "P":
                        column_name = 'Pitch range (P)'
                    case _:
                        raise ValueError(f"Unknown intensity character '{intensity_char}' in instrument '{instrument}'")
                    
                # Index of column column_name
                column_index = instrument_metadata.columns.get_loc(column_name)

                pitch_range = row.iloc[column_index]
                if pd.isna(pitch_range) or pitch_range == "nan":
                    continue
                
                if ">>" in pitch_range:
                    min_pitch = pitch_range[:3]
                    if "#" not in min_pitch and "b" not in min_pitch:
                        min_pitch = pitch_range[:2]
                    max_pitch = pitch_range[-3:]
                    if "#" not in max_pitch and "b" not in max_pitch:
                        max_pitch = pitch_range[-2:]
                else:
                    min_pitch = pitch_range[0] + '2'
                    max_pitch = pitch_range[0] + '2'

                min_midis.append(note_to_midi(min_pitch))
                max_midis.append(note_to_midi(max_pitch))
    
    # Return the min and max MIDI values for the sample
    return (min(min_midis), max(max_midis))

instruments_names = {
    "ALTO SAX": ["ALTO SAX"],
    "BARITONE SAX": ["BARITONE SAX"],
    "BASSOON": ["BASSOON (FAGOTTO)"],
    "CELLO": ["CELLO"],
    "CLARINET": ["CLARINET"],
    "CONTRABASS": ["CONTRABASS (WOOD BASS)"],
    "ENGLISH HORN": ["ENGLISH HORN"],
    "FLUTE": ["FLUTE"],
    "GLOCKENSPIEL": ["GLOCKENSPIEL"],
    "HARP": ["HARP"],
    "HARPSICHORD": ["HARPSICHORD (CEMBALO)"],
    "HORN": ["HORN"],
    "MARIMBA": ["MARIMBA"],
    "OBOE": ["OBOE"],
    "PIANOFORTE": ["PIANOFORTE"],
    "PICCOLO": ["PICCOLO"],
    "SOPRANO SAX": ["SOPRANO SAX"],
    "TENOR SAX": ["TENOR SAX"],
    "TIMPANI": [
        "TIMPANI 1 (23 inches)",
        "TIMPANI 2 (26 inches)",
        "TIMPANI 3 (29 inches)",
        "TIMPANI 4 (32 inches)"
        ],
    "TROMBONE": ["TROMBONE"],
    "TRUMPET": ["TRUMPET"],
    "TUBA": ["TUBA"],
    "VIBRAPHONE": ["VIBRAPHONE"],
    "VIOLA": ["VIOLA"],
    "VIOLIN": ["VIOLIN"],
    "XYLOPHONE": ["XYLOPHONE"]
}

instruments_prompts = {
    "ALTO SAX": "Alto Saxophone",
    "BARITONE SAX": "Baritone Saxophone",
    "BASS DRUM": "Bass Drum",
    "BASSOON": "Bassoon",
    "CELLO": "Cello",
    "CLARINET": "Clarinet",
    "CONTRABASS": "Contrabass",
    "CRASH CYMBAL": "Crash Cymbal",
    "ENGLISH HORN": "English Horn",
    "FLUTE": "Flute",
    "GLOCKENSPIEL": "Glockenspiel",
    "HARP": "HARP",
    "HARPSICHORD": "Harpsichord",
    "HORN": "Horn",
    "MARIMBA": "Marimba",
    "OBOE": "Oboe",
    "PIANOFORTE": "Piano",
    "PICCOLO": "Piccolo",
    "SNARE DRUM": "Snare Drum",
    "SOPRANO SAX": "Soprano Saxophone",
    "TENOR SAX": "Tenor Saxophone",
    "TIMPANI": "Timpani",
    "TRIANGLE": "Triangle",
    "TROMBONE": "Trombone",
    "TRUMPET": "Trumpet",
    "TUBA": "Tuba",
    "VIBRAPHONE": "Vibraphone",
    "VIOLA": "Viola",
    "VIOLIN": "Violin",
    "WOOD BLOCK": "Wood Block",
    "XYLOPHONE": "Xylophone"
}

instruments_caps_locked_prompts = {
    "ALTO SAX": "ALTO SAXOPHONE",
    "BARITONE SAX": "BARITONE SAXOPHONE",
    "BASS DRUM": "BASS DRUM",
    "BASSOON": "BASSOON",
    "CELLO": "CELLO",
    "CLARINET": "CLARINET",
    "CONTRABASS": "CONTRABASS",
    "CRASH CYMBAL": "CRASH CYMBAL",
    "ENGLISH HORN": "ENGLISH HORN",
    "FLUTE": "FLUTE",
    "GLOCKENSPIEL": "GLOCKENSPIEL",
    "HARP": "HARP",
    "HARPSICHORD": "HARPSICHORD",
    "HORN": "HORN",
    "MARIMBA": "MARIMBA",
    "OBOE": "OBOE",
    "PIANOFORTE": "PIANO",
    "PICCOLO": "PICCOLO",
    "SNARE DRUM": "SNARE DRUM",
    "SOPRANO SAX": "SOPRANO SAXOPHONE",
    "TENOR SAX": "TENOR SAXOPHONE",
    "TIMPANI": "TIMPANI",
    "TRIANGLE": "TRIANGLE",
    "TROMBONE": "TROMBONE",
    "TRUMPET": "TRUMPET",
    "TUBA": "TUBA",
    "VIBRAPHONE": "VIBRAPHONE",
    "VIOLA": "VIOLA",
    "VIOLIN": "VIOLIN",
    "WOOD BLOCK": "WOOD BLOCK",
    "XYLOPHONE": "XYLOPHONE"
}