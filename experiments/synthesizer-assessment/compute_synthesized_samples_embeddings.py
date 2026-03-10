import os
import pandas as pd
from tqdm import tqdm
import torch
import audio_to_embedding_tensor as atct

def compute_embeddings(condition_types: list[str]):
    for condition_type in condition_types:
        synth_metadata_path = f"resources/metadata/Synth/{condition_type}_metadata.csv"
        synth_metadata = pd.read_csv(synth_metadata_path)

        samples_paths = []
        save_paths = []
        for path, instrument in zip(synth_metadata["Path"], synth_metadata["Instrument"]):
            samples_paths.append(path)
            file_name = os.path.basename(path)
            file_name = file_name.replace('.wav', '')
            save_paths.append(f"{file_name}_embedding.pt")

        save_dir = os.path.join("resources/Synth/Embeddings", f"{condition_type}_embeddings")
        os.path.exists(save_dir) or os.makedirs(save_dir, exist_ok=True)
        atc = atct.Audio_to_Embedding_Tensor(embedding_type="clap")
        audios = atc.load_all_audios(samples_paths, crop_to_duration=5.0, pad_to_duration=5.0)
        for indice, audio in tqdm(enumerate(audios), total=len(audios), desc=f"Computing {atc.embedding_type} embeddings"):
            if os.path.exists(os.path.join("resources/Synth/Embeddings", f"{condition_type}_embeddings", save_paths[indice])):
                continue
            embedding = atc.get_embedding(audio)
            torch.save(embedding.cpu(), os.path.join("resources/Synth/Embeddings", f"{condition_type}_embeddings", save_paths[indice]))