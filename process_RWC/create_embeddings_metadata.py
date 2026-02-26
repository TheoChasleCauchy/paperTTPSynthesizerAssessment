import pandas as pd
import os

def create_embeddings_metadata():
    for embedding_type in ["clap", "clap-music", "mert", "vggish"]:
        samples_metadata = "data/RWC/RWC_metadata.csv"

        # Read the samples metadata CSV file
        samples_df = pd.read_csv(samples_metadata)

        # Create a new DataFrame for embeddings metadata
        embeddings_metadata = samples_df.copy()

        # Add a column for embedding file paths
        embeddings_metadata["Path"] = embeddings_metadata["Path"].apply(lambda x: f"data/RWC/embeddings/{embeddings_type}_embeddings/{x.replace('/', '_').replace('.wav', '')}_embedding.pt")

        # Save the result to a new CSV file
        output_dir = f"data/RWC/metadata/{embeddings_type}"
        os.makedirs(output_dir, exist_ok=True)
        embeddings_metadata.to_csv(f"{output_dir}/{embeddings_type}_embeddings_labels.csv", index=False)

        print(f"Embeddings metadata file saved as '{output_dir}/{embeddings_type}_embeddings_labels.csv'")
