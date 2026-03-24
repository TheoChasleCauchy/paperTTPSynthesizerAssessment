import audio_to_embedding_tensor as atct  # Custom module for converting audio to embeddings
import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces for directory and file operations
from tqdm import tqdm  # Progress bar for iterative tasks
import torch  # PyTorch for tensor operations and saving embeddings

def compute_embeddings():
    """
    Compute embeddings for audio samples in the RWC dataset using various embedding models.

    This function:
    1. Loads the RWC metadata CSV file containing paths to audio samples.
    2. Constructs full paths for audio samples and corresponding paths for saving embeddings.
    3. Computes embeddings for each audio sample using multiple embedding models (CLAP, CLAP-Music, MERT, VGGish).
    4. Saves the computed embeddings to disk, skipping already computed embeddings.

    Steps:
    - Load the RWC metadata CSV file.
    - Construct full paths for audio samples and embedding save paths.
    - For each embedding type, compute and save embeddings for all audio samples.
    - Skip computation if the embedding file already exists.

    Returns:
        None: Embeddings are saved to disk in the specified directory structure.
    """
    print("[INFO] Computing embeddings for RWC samples.")

    # Path to the RWC metadata CSV file
    RWC_metadata_path = "data/metadata/RWC/RWC_metadata.csv"

    # Load the RWC metadata
    RWC_metadata = pd.read_csv(RWC_metadata_path)

    # Initialize lists to store full paths for audio samples and save paths for embeddings
    samples_paths = []
    save_paths = []

    # Construct full paths for audio samples and corresponding save paths for embeddings
    for path in RWC_metadata["Path"]:
        # Construct the full path to the audio sample
        full_path = f"data/RWC/RWC-preprocessed/{path}"
        samples_paths.append(full_path)

        # Construct the save path for the embedding, replacing slashes and removing the .wav extension
        path = path.replace('/', '_').replace('.wav', '')
        save_paths.append(f"{path}_embedding.pt")

    # Compute embeddings for each embedding type
    for embeddings_type in ["clap", "clap-music", "mert", "vggish"]:
        # Create the directory for saving embeddings of the current type
        save_dir = os.path.join("data/RWC/embeddings/", f"{embeddings_type}_embeddings")
        os.makedirs(save_dir, exist_ok=True)

        # Initialize the audio-to-embedding converter with the current embedding type
        atc = atct.Audio_to_Embedding_Tensor(embedding_type=embeddings_type)

        # Load all audio samples, cropping and padding to 5 seconds
        audios = atc.load_all_audios(samples_paths, crop_to_duration=5.0, pad_to_duration=5.0)

        # Iterate over each audio sample and compute its embedding
        for indice, audio in tqdm(enumerate(audios), total=len(audios), desc=f"Computing {atc.embedding_type} embeddings"):
            # Skip if the embedding file already exists
            if os.path.exists(os.path.join(save_dir, save_paths[indice])):
                continue

            # Compute the embedding for the current audio sample
            embedding = atc.get_embedding(audio)

            # Save the embedding to disk
            torch.save(embedding.cpu(), os.path.join(save_dir, save_paths[indice]))
