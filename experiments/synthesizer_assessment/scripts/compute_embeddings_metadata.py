import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces for directory and file operations

def compute_synthesized_samples_embeddings_metadata(embeddings_type: str):
    """
    Generate metadata for embeddings of synthesized audio samples, including quality traits from ground truth data.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").

    This function:
    1. Iterates over each condition type (text, audio, text_audio).
    2. Loads the metadata for synthesized audio samples.
    3. Constructs paths for embedding files.
    4. Loads the ground truth quality traits for each instrument.
    5. Maps the ground truth quality traits to the synthesized samples based on instrument.
    6. Saves the resulting metadata, including quality traits, to a CSV file.

    Steps:
    - For each condition type, load the synthesized samples metadata.
    - Construct paths for embedding files.
    - Load the ground truth quality traits.
    - Map quality traits to synthesized samples based on instrument.
    - Save the metadata, including quality traits, to a CSV file.

    Returns:
        None: Metadata files are saved to disk.
    """
    # List of condition types to process
    condition_types = ["text", "audio", "text_audio"]

    # Iterate over each condition type
    for condition_type in condition_types:
        # Path to the synthesized samples metadata CSV file
        samples_metadata = f"data/TokenSynth/Samples/{condition_type}_conditioned_synthesis/{condition_type}_conditioned_synthesis_metadata.csv"

        # Load the synthesized samples metadata
        samples_df = pd.read_csv(samples_metadata)

        # Create a new DataFrame for embeddings metadata by copying the samples metadata
        embeddings_metadata = samples_df.copy()

        # Construct paths for embedding files
        embeddings_dir = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}_embeddings"
        embeddings_metadata["Path"] = embeddings_metadata["Path"].apply(
            lambda x: f"{embeddings_dir}/{os.path.basename(x).replace('/', '_').replace('.wav', '')}_embedding.pt"
        )

        # Path to the ground truth quality traits CSV file
        ground_truth_path = f"data/Reymore/timbre_traits_ground_truth.csv"

        # Load the ground truth quality traits
        ground_truth_df = pd.read_csv(ground_truth_path)

        # Create a mapping from RWC Name to quality values
        ground_truth_mapping = {}
        for _, row in ground_truth_df.iterrows():
            # Skip the first two columns (Instrument, RWC Name) and map the rest to RWC Name
            ground_truth_mapping[row["RWC Name"]] = row[2:]

        # Add quality columns to embeddings metadata
        for quality in ground_truth_df.columns[2:]:
            embeddings_metadata[quality] = None

        # Populate quality values based on Instrument
        for idx, row in embeddings_metadata.iterrows():
            instrument = row["Instrument"]
            if instrument in ground_truth_mapping:
                for quality, value in zip(ground_truth_df.columns[2:], ground_truth_mapping[instrument]):
                    embeddings_metadata.at[idx, quality] = value

        # Path to the output embeddings metadata CSV file
        embeddings_metadata_path = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}_embeddings/embeddings_labels.csv"

        # Save the embeddings metadata to a CSV file
        embeddings_metadata.to_csv(embeddings_metadata_path, index=False)

        # Print a message indicating the location of the saved metadata file
        print(f"Embeddings metadata file saved as '{embeddings_metadata_path}'")
