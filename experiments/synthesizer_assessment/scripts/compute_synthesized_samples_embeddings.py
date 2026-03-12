import os
import pandas as pd
from tqdm import tqdm
import torch
import audio_to_embedding_tensor as atct

def compute_embeddings(embeddings_type: str):
    condition_types = ["text_conditioned", "audio_conditioned", "text_audio_conditioned"]
    for condition_type in condition_types:
        synth_metadata_path = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/{condition_type}_metadata.csv"
        synth_metadata = pd.read_csv(synth_metadata_path)

        samples_paths = []
        save_paths = []
        for path in synth_metadata["Path"]:
            samples_paths.append(path)
            file_name = os.path.basename(path)
            file_name = file_name.replace('.wav', '')
            save_paths.append(f"{file_name}_embedding.pt")

        save_dir = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}_embeddings/"
        os.path.exists(save_dir) or os.makedirs(save_dir, exist_ok=True)
        atc = atct.Audio_to_Embedding_Tensor(embedding_type=embeddings_type)
        audios = atc.load_all_audios(samples_paths, crop_to_duration=5.0, pad_to_duration=5.0)
        for indice, audio in tqdm(enumerate(audios), total=len(audios), desc=f"Computing {atc.embedding_type} embeddings"):
            if os.path.exists(os.path.join(save_dir, save_paths[indice])):
                continue
            embedding = atc.get_embedding(audio)
            torch.save(embedding.cpu(), os.path.join(save_dir, save_paths[indice]))