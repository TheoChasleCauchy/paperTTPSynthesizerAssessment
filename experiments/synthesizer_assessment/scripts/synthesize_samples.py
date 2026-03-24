from tokensynth import TokenSynth, CLAP, DACDecoder  # TokenSynth, CLAP, and DACDecoder models
import audiofile  # Library for reading and writing audio files
import torch  # PyTorch for tensor operations and device management
import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bar for iterative tasks
import os  # Operating system interfaces for directory and file operations
from utils import get_midi_range_from_instrument, instruments_caps_locked_prompts, midi_to_note  # Utility functions

def synthesize_audios(condition_type: str, seed: int):
    """
    Synthesize audio samples for each instrument using TokenSynth, conditioned on text, audio, or both.

    Args:
        condition_type (str): Type of conditioning for synthesis. Must be one of "text", "audio", or "text_audio".
        seed (int): Random seed for reproducibility.

    Raises:
        AssertionError: If `condition_type` is not one of the allowed values.

    This function:
    1. Initializes the TokenSynth, CLAP, and DACDecoder models.
    2. Loads the ground truth timbre traits for instruments.
    3. For each instrument, generates 100 random MIDI notes within its range.
    4. For each note, synthesizes audio using the specified condition type.
    5. Saves the synthesized audio to disk.

    Steps:
    - Initialize models and set the device.
    - Load the ground truth timbre traits.
    - For each instrument, generate random notes or load previously generated notes.
    - For each note, synthesize audio using the specified condition type.
    - Save the synthesized audio to disk.

    Returns:
        None: Synthesized audio files are saved to disk.
    """
    # Validate the condition type
    assert condition_type in ["text", "audio", "text_audio"], f"Invalid condition type: {condition_type}"

    # Initialize models and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    # Load the ground truth timbre traits
    timbre_traits_ground_truth = pd.read_csv("data/Reymore/timbre_traits_ground_truth.csv")
    instruments = timbre_traits_ground_truth["RWC Name"].unique()

    # Create the output directory for synthesized audio
    output_dir = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each instrument
    for instrument in instruments:
        # Create a subdirectory for the instrument if it doesn't exist
        os.makedirs(f"{output_dir}/{instrument}", exist_ok=True)

        # Load or generate random notes for the instrument
        if os.path.exists(f"{output_dir}/{instrument}/random_notes.txt"):
            # Load previously generated random notes
            with open(f"{output_dir}/{instrument}/random_notes.txt", "r") as f:
                random_notes = [int(line.strip()) for line in f.readlines()]
        else:
            # Get the MIDI note range for the instrument
            midi_range = get_midi_range_from_instrument(instrument)

            # Sample 100 notes in the range with a Gaussian distribution
            mean = (midi_range[0] + midi_range[1]) // 2
            std = 20/100 * (midi_range[1] - midi_range[0])
            random_notes = torch.normal(mean, std, size=(100,), generator=torch.Generator().manual_seed(seed)).int().tolist()

            # Save the generated random notes to a file
            with open(f"{output_dir}/{instrument}/random_notes.txt", "w") as f:
                f.write("\n".join(map(str, random_notes)))

        # Iterate over each random note and synthesize audio
        for i, midi in tqdm(enumerate(random_notes), total=len(random_notes), desc=f"Synthesizing {instrument} from {condition_type} condition"):
            # Define the output file path
            output_file = f"{output_dir}/{instrument}/{instrument}_sample_{i+1}_note_{midi_to_note(midi)}.wav"

            # Skip if the output file already exists
            if os.path.exists(output_file):
                continue

            # Load the MIDI file for the note
            midi = f"data/TokenSynth/midi_files/input_midi_{midi}.mid"

            # Disable gradient computation for efficiency
            with torch.no_grad():
                # Get the condition based on the condition type
                match condition_type:
                    case "text":
                        # Encode the instrument's text description as the condition
                        condition = clap.encode_text(instruments_caps_locked_prompts[instrument])

                    case "audio":
                        # Load the mean CLAP embedding for the instrument as the condition
                        embeddings_path = f"data/RWC/mean_clap_embeddings/"
                        condition = torch.load(f"{embeddings_path}{instrument}_embedding.pt")

                    case "text_audio":
                        # Combine text and audio conditions by averaging their embeddings
                        text_condition = clap.encode_text(instruments_caps_locked_prompts[instrument])
                        embeddings_path = f"data/RWC/mean_clap_embeddings/"
                        audio_condition = torch.load(f"{embeddings_path}{instrument}_embedding.pt")
                        condition = torch.mean(torch.stack([text_condition, audio_condition]), dim=0)

                    case _:
                        raise ValueError(f"Invalid condition type: {condition_type}")

                # Synthesize audio tokens using the condition and MIDI file
                tokens_audio = synth.synthesize(condition, midi, top_k=10)

                # Decode the audio tokens into a waveform
                audio_audio = decoder.decode(tokens_audio)

            # Save the synthesized audio to disk
            audiofile.write(output_file, audio_audio.cpu().numpy(), 16000)

def synthesize_all(seed: int):
    """
    Synthesize audio samples for all condition types (text, audio, text_audio).

    Args:
        seed (int): Random seed for reproducibility.

    This function calls `synthesize_audios` for each condition type.

    Returns:
        None: Synthesized audio files are saved to disk.
    """
    # Synthesize audio for each condition type
    synthesize_audios("text", seed)
    synthesize_audios("audio", seed)
    synthesize_audios("text_audio", seed)
