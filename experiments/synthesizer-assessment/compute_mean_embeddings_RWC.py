from tokensynth import CLAP
import pandas as pd
from tqdm import tqdm
import os
import torch

def compute_mean_embeddings():
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap = CLAP(device=device)

    metadata_path = "data/metadata/RWC/RWC_metadata.csv"
    audios_path = "data/RWC/RWC-preprocessed/Dataset/"
    embeddings_path = "data/RWC/mean_embeddings/clap_embeddings/"
    os.path.exists(embeddings_path) or os.makedirs(embeddings_path, exist_ok=True)

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Group metadata by instruments
    instrument_groups = metadata.groupby("Instrument")
    for instrument, instrument_df in instrument_groups:
        if os.path.exists(f"{embeddings_path}{instrument}_embedding.pt"):
            continue
        with torch.no_grad():
            embeddings = []
            for _, row in tqdm(instrument_df.iterrows(), total=len(instrument_df), desc=f"Processing mean embeddings for instrument {instrument}"):
                timbre_audio_i = clap.encode_audio(os.path.join(audios_path, row["Path"]))
                embeddings.append(timbre_audio_i)
            audio_embedding = torch.mean(torch.stack(embeddings), dim=0)
            torch.save(audio_embedding, f"{embeddings_path}{instrument}_embedding.pt")