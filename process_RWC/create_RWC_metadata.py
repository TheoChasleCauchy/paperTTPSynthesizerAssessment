import pandas as pd
import os
from tqdm import tqdm

def create_RWC_metadata():
    qualities_ground_truth = pd.read_csv("data/Reymore/timber_traits_ground_truth.csv")
    unique_instruments = qualities_ground_truth["RWC Name"].unique()
    qualities_names = qualities_ground_truth.columns[2:]  # Skip "RWC Name" and "Instrument" columns

    rwc_dataset_path = "data/RWC/RWC_preprocessed"

    metadata = {"Path": [], "Instrument": [], **{quality: [] for quality in qualities_names}}

    # Get all the files paths for each instrument
    for instrument in unique_instruments:
        instrument_path = os.path.join(rwc_dataset_path, instrument)
        if os.path.exists(instrument_path):
            files = os.listdir(instrument_path)
            for file in tqdm(files, total=len(files), desc=f"Processing {instrument}"):
                metadata["Path"].append(os.path.join(instrument, file))
                metadata["Instrument"].append(instrument)
                for quality in qualities_names:
                    metadata[quality].append(qualities_ground_truth[qualities_ground_truth["RWC Name"] == instrument][quality].iloc[0])

    # Convert metadata to a dataframe to save it as a CSV file
    metadata_df = pd.DataFrame(metadata)
    metadata_path = "data/RWC/metadata"
    os.path.exists(metadata_path) or os.makedirs(metadata_path)
    metadata_df.to_csv(os.path.join(metadata_path, "RWC_metadata.csv"), index=False)