import audio_to_embedding_tensor as atct  # Custom module for converting audio files to embeddings
import pandas as pd  # Library for data manipulation and analysis
import os  # Library for interacting with the operating system (directory and file operations)
from tqdm import tqdm  # Library for displaying progress bars during iterative tasks
import torch  # PyTorch library for tensor operations and saving embeddings

def compute_embeddings(audios_folder: str):
    """
    Traverses a directory, applies an embedding function to each audio file, and saves the results in a CSV file.

    Args:
        audios_folder (str): Path to the directory containing audio files.
    """

    # Initialize the audio-to-embedding converter with the specified embedding type ("clap" in this case)
    atc = atct.Audio_to_Embedding_Tensor(embedding_type="clap")

    # Supported audio file extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}

    # List to store the results (audio paths and corresponding embedding paths)
    embeddings = []

    # Traverse the directory and process each audio file
    for root, _, files in tqdm(os.walk(audios_folder), desc="Computing audio embeddings"):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Check if the file is an audio file based on its extension
            if file_ext in audio_extensions:
                # Load the audio file
                audio_file = atc.load_audio(file_path)

                # Compute the embedding for the current audio file
                embedding = atc.get_embedding(audio_file)

                # Create a directory to save embeddings (mirroring the original audio directory structure)
                embeddings_folder = os.path.join("data", os.path.dirname(file_path) + "_embeddings")
                os.makedirs(embeddings_folder, exist_ok=True)

                # Define the path for the embedding file (replace audio extension with .pt)
                embedding_path = os.path.join(
                    embeddings_folder,
                    os.path.basename(file_path).replace(os.path.splitext(file_path)[1], ".pt")
                )

                # Save the embedding tensor to disk
                torch.save(embedding.cpu(), embedding_path)

                # Append the audio path and embedding path to the results list
                embeddings.append({"audio_path": file_path, "embedding_path": embedding_path})

    # Create a DataFrame from the results and save it as a CSV file
    df = pd.DataFrame(embeddings)
    output_csv_path = os.path.join("inference", "results", os.path.basename(audios_folder), "inference_results.csv")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # Ensure the output directory exists
    df.to_csv(output_csv_path, index=False)
