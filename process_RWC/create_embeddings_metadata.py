import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces for directory and file operations

def create_embeddings_metadata():
    """
    Create metadata for embeddings generated from the RWC dataset.

    This function:
    1. Iterates over each embedding type (CLAP, CLAP-Music, MERT, VGGish).
    2. Loads the RWC metadata CSV file containing information about audio samples.
    3. Creates a new DataFrame for embeddings metadata by copying the samples metadata.
    4. Updates the "Path" column to point to the corresponding embedding files.
    5. Saves the embeddings metadata to a new CSV file for each embedding type.

    Steps:
    - For each embedding type, load the RWC metadata.
    - Copy the samples metadata to a new DataFrame for embeddings.
    - Update the "Path" column to reflect the location of the embedding files.
    - Save the embeddings metadata to a CSV file in a dedicated directory.

    Returns:
        None: Metadata files are saved to disk in the specified directory structure.
    """
    print("[INFO] Creating embeddings metadata.")

    # Iterate over each embedding type
    for embedding_type in ["clap", "clap-music", "mert", "vggish"]:
        # Path to the RWC samples metadata CSV file
        samples_metadata = "data/metadata/RWC/RWC_metadata.csv"

        # Load the samples metadata CSV file
        samples_df = pd.read_csv(samples_metadata)

        # Create a new DataFrame for embeddings metadata by copying the samples metadata
        embeddings_metadata = samples_df.copy()

        # Update the "Path" column to point to the corresponding embedding files
        embeddings_metadata["Path"] = embeddings_metadata["Path"].apply(
            lambda x: f"data/RWC/embeddings/{embedding_type}_embeddings/{x.replace('/', '_').replace('.wav', '')}_embedding.pt"
        )

        # Create the output directory for the embeddings metadata
        output_dir = f"data/metadata/RWC/{embedding_type}_embeddings"
        os.makedirs(output_dir, exist_ok=True)

        # Save the embeddings metadata to a CSV file
        embeddings_metadata.to_csv(
            f"{output_dir}/{embedding_type}_embeddings_labels.csv",
            index=False
        )

        print(f"Embeddings metadata file saved as '{output_dir}/{embedding_type}_embeddings_labels.csv'")
