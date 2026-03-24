import pandas as pd
import os
from tqdm import tqdm  # Progress bar for iterative tasks

def create_RWC_metadata():
    """
    Create metadata for the RWC (Real World Computing) instrument dataset.

    This function reads ground truth timbre quality traits from a CSV file,
    processes the RWC preprocessed dataset directory structure, and generates
    a comprehensive metadata CSV file. The metadata includes file paths, instrument names,
    and quality trait values for each audio file.

    Steps:
    1. Loads timbre quality ground truth data from the Reymore dataset.
    2. Extracts unique instrument names and quality trait column names.
    3. Iterates through the RWC preprocessed dataset directory.
    4. For each audio file, collects its path, instrument name, and corresponding quality traits.
    5. Combines all metadata into a pandas DataFrame.
    6. Saves the resulting metadata to a CSV file.

    Returns:
        None: The function writes the metadata to a CSV file at `data/metadata/RWC/RWC_metadata.csv`.

    Raises:
        FileNotFoundError: If the ground truth CSV or RWC dataset path does not exist.
    """
    print("[INFO] Creating RWC metadata.")

    # Load the ground truth timbre quality traits from the Reymore dataset
    qualities_ground_truth = pd.read_csv("data/Reymore/timbre_traits_ground_truth.csv")

    # Extract unique instrument names from the ground truth data
    unique_instruments = qualities_ground_truth["RWC Name"].unique()

    # Extract quality trait column names, skipping the first two columns ("RWC Name" and "Instrument")
    qualities_names = qualities_ground_truth.columns[2:]

    # Path to the RWC preprocessed dataset directory
    rwc_dataset_path = "data/RWC/RWC-preprocessed"

    # Initialize metadata dictionary with keys for path, instrument, and each quality trait
    metadata = {
        "Path": [],
        "Instrument": [],
        **{quality: [] for quality in qualities_names}
    }

    # Iterate over each unique instrument
    for instrument in unique_instruments:
        # Construct the path to the instrument's directory
        instrument_path = os.path.join(rwc_dataset_path, instrument)

        # Check if the instrument directory exists
        if os.path.exists(instrument_path):
            # List all files in the instrument directory
            files = os.listdir(instrument_path)

            # Process each file in the instrument directory
            for file in tqdm(files, total=len(files), desc=f"Processing {instrument}"):
                # Append the relative path of the file to the metadata
                metadata["Path"].append(os.path.join(instrument, file))

                # Append the instrument name to the metadata
                metadata["Instrument"].append(instrument)

                # For each quality trait, append the corresponding value from the ground truth data
                for quality in qualities_names:
                    # Retrieve the quality trait value for the current instrument
                    quality_value = qualities_ground_truth[
                        qualities_ground_truth["RWC Name"] == instrument
                    ][quality].iloc[0]
                    metadata[quality].append(quality_value)

    # Convert the metadata dictionary to a pandas DataFrame
    metadata_df = pd.DataFrame(metadata)

    # Create the output directory for the metadata if it doesn't exist
    metadata_path = "data/metadata/RWC"
    os.makedirs(metadata_path, exist_ok=True)

    # Save the metadata DataFrame to a CSV file
    metadata_df.to_csv(os.path.join(metadata_path, "RWC_metadata.csv"), index=False)
