import audio_to_embedding_tensor as atct
import pandas as pd
import os
from tqdm import tqdm
import torch

RWC_metadata_path = "data/RWC/metadata/RWC_metadata.csv"

RWC_metadata = pd.read_csv(RWC_metadata_path)

def compute_embeddings():

    # Get audios paths and embeddings saving paths
    samples_paths = []
    save_paths = []
    for path, instrument in zip(RWC_metadata["Path"], RWC_metadata["Instrument"]):
        full_path = f"./resources/RWC/Dataset/{path}"
        samples_paths.append(full_path)
        path = path.replace('/', '_').replace('.wav', '')
        save_paths.append(f"{path}_embedding.pt")
    
    # Compute non-existant embeddings
    for embeddings_type in ["clap", "clap-music", "mert", "vggish"]:
        save_dir = os.path.join("data/RWC/embeddings/", f"{embeddings_type}_embeddings")
        os.path.exists(save_dir) or os.makedirs(save_dir, exist_ok=True)
        atc = atct.Audio_to_Embedding_Tensor(embedding_type=embeddings_type)
        audios = atc.load_all_audios(samples_paths, crop_to_duration=5.0, pad_to_duration=5.0)
        for indice, audio in tqdm(enumerate(audios), total=len(audios), desc=f"Computing {atc.embedding_type} embeddings"):
            if os.path.exists(os.path.join(save_dir, save_paths[indice])):
                continue
            embedding = atc.get_embedding(audio)
            torch.save(embedding.cpu(), os.path.join(save_dir, save_paths[indice]))
