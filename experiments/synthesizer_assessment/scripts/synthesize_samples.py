from tokensynth import TokenSynth, CLAP, DACDecoder
import audiofile
import torch
import pandas as pd
from tqdm import tqdm
import os
from utils import get_midi_range_from_instrument, instruments_caps_locked_prompts, midi_to_note

def synthesize_audios(condition_type: str, seed: int):

    assert condition_type in ["text", "audio", "text_audio"], f"Invalid condition type: {condition_type}"

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)

    timbre_traits_ground_truth = pd.read_csv("data/Reymore/timber_traits_ground_truth.csv")
    instruments = timbre_traits_ground_truth["RWC Name"].unique()

    output_dir = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/"
    os.makedirs(output_dir, exist_ok=True)

    for instrument in instruments:

        os.path.exists(f"{output_dir}/{instrument}") or os.makedirs(f"{output_dir}/{instrument}", exist_ok=True)

        # Get the random notes or sample them
        if os.path.exists(f"{output_dir}/{instrument}/random_notes.txt"):
            with open(f"{output_dir}/{instrument}/random_notes.txt", "r") as f:
                random_notes = [int(line.strip()) for line in f.readlines()]
        else:
            midi_range = get_midi_range_from_instrument(instrument) # get the midi notes range for this audio

            # Sample 100 notes in the range with a gaussian distribution centered on the mean of the range and a std of 20% of the range length
            mean = (midi_range[0] + midi_range[1]) // 2
            std = 20/100 * (midi_range[1] - midi_range[0])
            random_notes = torch.normal(mean, std, size=(100,), generator=torch.Generator().manual_seed(seed)).int().tolist()

            # Save the random notes to a file
            with open(f"{output_dir}/{instrument}/random_notes.txt", "w") as f:
                f.write("\n".join(map(str, random_notes)))

        for i, midi in tqdm(enumerate(random_notes), total=len(random_notes), desc=f"Synthesizing {instrument} from {condition_type} condition"):  # C4 to B4
            
            output_file = f"{output_dir}/{instrument}/{instrument}_sample_{i+1}_note_{midi_to_note(midi)}.wav"
            if os.path.exists(output_file):
                continue

            midi = f"data/TokenSynth/midi_files/input_midi_{midi}.mid"
            
            with torch.no_grad():
                # Get the condition:
                match condition_type:
                    case "text":
                        condition = clap.encode_text(instruments_caps_locked_prompts[instrument])

                    case "audio":
                        embeddings_path = f"data/RWC/mean_clap_embeddings/"
                        condition = torch.load(f"{embeddings_path}{instrument}_embedding.pt")

                    case "text_audio":
                        text_condition = clap.encode_text(instruments_caps_locked_prompts[instrument])
                        embeddings_path = f"data/RWC/mean_clap_embeddings/"
                        audio_condition = torch.load(f"{embeddings_path}{instrument}_embedding.pt")
                        condition = torch.mean(torch.stack([text_condition, audio_condition]), dim=0)

                    case _:
                        raise ValueError(f"Invalid condition type: {condition_type}")

                # Generate audio tokens
                tokens_audio = synth.synthesize(condition, midi, top_k=10)

                # Decode tokens into audio waveforms
                audio_audio = decoder.decode(tokens_audio) 

            # Save audio files
            audiofile.write(output_file, audio_audio.cpu().numpy(), 16000)

def synthesize_all(seed: int):
    
    synthesize_audios("text", seed)
    synthesize_audios("audio", seed)
    synthesize_audios("text_audio", seed)