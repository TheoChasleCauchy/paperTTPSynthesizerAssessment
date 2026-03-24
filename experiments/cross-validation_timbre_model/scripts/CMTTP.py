import pandas as pd  # Data manipulation and analysis
import laion_clap  # CLAP model for audio and text embeddings
import torch  # PyTorch for tensor operations and device management
from tqdm import tqdm  # Progress bar for iterative tasks
import numpy as np  # Numerical operations
import os  # Operating system interfaces for directory and file operations

def CMTTP():
    """
    Compute and evaluate Cross-Modal Timbre Trait Prediction (CMTTP) using CLAP embeddings.

    This function:
    1. Loads timbre trait labels and splits trait names with hyphens.
    2. Loads the CLAP model and computes text embeddings for each timbre trait.
    3. Loads RWC metadata and computes the distance between each sample's embedding and each trait embedding.
    4. Normalizes distances and refactors the DataFrame to keep only the minimum distance for each trait couple.
    5. Inverts distances to get scores and computes absolute errors.
    6. Computes and saves the Mean Absolute Error (MAE) for each instrument and trait.

    Steps:
    - Load timbre trait labels and split trait names with hyphens.
    - Load the CLAP model and compute text embeddings for each timbre trait.
    - Load RWC metadata and compute distances between sample embeddings and trait embeddings.
    - Normalize distances and refactor the DataFrame.
    - Invert distances to get scores and compute absolute errors.
    - Compute and save the MAE for each instrument and trait.

    Returns:
        None: Results are saved to CSV files.
    """
    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load timbre trait labels
    timbre_traits_labels_path = "data/Reymore/timbre_traits_ground_truth.csv"
    timbre_traits_df = pd.read_csv(timbre_traits_labels_path)
    timbre_traits_names = timbre_traits_df.columns[2:].tolist()  # Get the names of the timbre traits (excluding the first two columns)

    # Split trait names with hyphens into tuples
    timbre_traits_tuples = []
    for trait in timbre_traits_names:
        if "-" in trait:
            left, right = trait.split("-", 1)
            timbre_traits_tuples.append([left.strip(), right.strip()])
        else:
            timbre_traits_tuples.append([trait])

    # Load the CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # Download the default pretrained checkpoint

    # Compute and save text embeddings for each timbre trait
    os.makedirs("data/CMTTP/timbre_traits_embeddings", exist_ok=True)
    for traits_couples in timbre_traits_tuples:
        for trait in traits_couples:
            text_embed = model.get_text_embedding(trait, use_tensor=True)
            torch.save(text_embed, f"data/CMTTP/timbre_traits_embeddings/{trait}.pt")

    # Load RWC metadata
    rwc_clap_embeddings_metadata_path = "data/metadata/RWC/clap_embeddings/clap_embeddings_labels.csv"
    rwc_clap_embeddings_metadata = pd.read_csv(rwc_clap_embeddings_metadata_path)

    # Initialize a DataFrame to store distances between sample embeddings and trait embeddings
    samples_traits_distances_df = rwc_clap_embeddings_metadata[rwc_clap_embeddings_metadata.columns[0:2]]  # Keep "Path" and "Instrument" columns

    # Compute distances between each sample embedding and each trait embedding
    for _, row in tqdm(samples_traits_distances_df.iterrows(), total=len(samples_traits_distances_df)):
        sample_embedding = torch.load(row["Path"], weights_only=True).to(device)
        for trait_tuple in timbre_traits_tuples:
            for trait in trait_tuple:
                trait_embedding = torch.load(f"data/CMTTP/timbre_traits_embeddings/{trait}.pt", weights_only=True).to(device)
                distance = torch.norm(sample_embedding - trait_embedding).item()
                samples_traits_distances_df.loc[samples_traits_distances_df["Path"] == row["Path"], trait] = distance

    # Save the distances DataFrame
    os.makedirs("models/cross-validation_timbre_model/CMTTP", exist_ok=True)
    samples_traits_distances_df.to_csv("models/cross-validation_timbre_model/CMTTP/samples_timbre_traits_distances.csv", index=False)

    # Normalize all distances by the maximum distance in the DataFrame
    samples_traits_distances_df[samples_traits_distances_df.columns[2:]] = samples_traits_distances_df[samples_traits_distances_df.columns[2:]].div(samples_traits_distances_df[samples_traits_distances_df.columns[2:]].max(axis=0), axis=1)
    samples_traits_distances_df.to_csv("models/cross-validation_timbre_model/CMTTP/samples_timbre_traits_distances.csv", index=False)

    # For each couple of traits, keep only the minimum value
    samples_traits_distances_refactored_df = samples_traits_distances_df.copy()
    for trait_tuple in timbre_traits_tuples:
        if len(trait_tuple) > 1:
            min_val = samples_traits_distances_refactored_df[trait_tuple].min(axis=1)
            for trait in trait_tuple:
                samples_traits_distances_refactored_df.drop(columns=[trait], inplace=True)
            samples_traits_distances_refactored_df[f"{'-'.join(trait_tuple)}"] = min_val

    # Save the refactored distances DataFrame
    samples_traits_distances_refactored_df.to_csv("models/cross-validation_timbre_model/CMTTP/samples_timbre_traits_distances_refactored.csv", index=False)

    # Invert distances to get scores
    samples_traits_inversed_distances_df = samples_traits_distances_refactored_df.copy()
    samples_traits_inversed_distances_df[samples_traits_inversed_distances_df.columns[2:]] = 1 - samples_traits_inversed_distances_df[samples_traits_inversed_distances_df.columns[2:]]
    samples_traits_inversed_distances_df.to_csv("models/cross-validation_timbre_model/CMTTP/CMTTP_predictions.csv", index=False)

    # Compute absolute errors for each trait and each sample
    samples_traits_absolute_errors_df = samples_traits_inversed_distances_df.copy()
    for index, row in tqdm(samples_traits_absolute_errors_df.iterrows(), total=len(samples_traits_absolute_errors_df)):
        instrument = row["Instrument"]
        for trait in timbre_traits_names:
            predicted_value = row[trait]
            ground_truth_row = timbre_traits_df.loc[timbre_traits_df["RWC Name"] == instrument]
            ground_truth_value = ground_truth_row[trait].iloc[0]
            # Normalize ground truth values
            ground_truth_value = (ground_truth_value - 1) / 6.0
            samples_traits_absolute_errors_df.loc[index, trait] = abs(predicted_value - ground_truth_value)

    # Save the absolute errors DataFrame
    samples_traits_absolute_errors_df.to_csv("models/cross-validation_timbre_model/CMTTP/CMTTP_absolute_errors.csv", index=False)

    # Compute average metrics for each instrument and trait
    average_metric = {}
    # Group data by instrument
    grouped_by_instrument = samples_traits_absolute_errors_df.groupby('Instrument')
    for instrument, instrument_df in grouped_by_instrument:
        average_metric[instrument] = {}
        to_be_averaged = []
        for trait in timbre_traits_names:
            average_metric[instrument][trait] = instrument_df[trait].mean()
            to_be_averaged.append(average_metric[instrument][trait])
        average_metric[instrument]["Average"] = sum(to_be_averaged) / len(to_be_averaged)

    # Add an "Average" row
    average_metric["Average"] = {}
    for trait in timbre_traits_names + ["Average"]:
        average_metric["Average"][trait] = np.mean([average_metric[instrument][trait] for instrument in average_metric.keys() if instrument != "Average"])

    # Convert the average metrics dictionary to a DataFrame
    df = pd.DataFrame.from_dict(average_metric, orient="index")

    # Place the "Average" column at the beginning of the DataFrame
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Average")))
    df = df[cols]

    # Save the MAE DataFrame
    df.to_csv("models/cross-validation_timbre_model/CMTTP/CMTTP_maes_per_instrument.csv", index=False)
