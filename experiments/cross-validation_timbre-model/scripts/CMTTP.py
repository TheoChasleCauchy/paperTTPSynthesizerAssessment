import logging

import pandas as pd
import laion_clap
import torch
from tqdm import tqdm
import numpy as np
import os

def CMTTP():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load timber traits
    timber_traits_labels_path = "data/Reymore/timber_traits_ground_truth.csv"
    timber_traits_df = pd.read_csv(timber_traits_labels_path)
    timber_traits_names = timber_traits_df.columns[2:].tolist() # Get the names of the timber_traits (excluding the 2 first columns)

    # For each timber trait, if it has "-" in it, split both sides into a list of 2
    timber_traits_tuples = []
    for trait in timber_traits_names:
        if "-" in trait:
            left, right = trait.split("-", 1)
            timber_traits_tuples.append([left.strip(), right.strip()])
        else:
            timber_traits_tuples.append([trait])

    # Load CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.
    
    # Compute and save all timber traits embeddings
    os.makedirs("data/CMTTP/timber_traits_embeddings", exist_ok=True)
    for traits_couples in timber_traits_tuples:
        for trait in traits_couples:
            text_embed = model.get_text_embedding(trait, use_tensor=True)
            torch.save(text_embed, f"data/CMTTP/timber_traits_embeddings/{trait}.pt")


    # Load RWC metadata
    rwc_clap_embeddings_metadata_path = "data/metadata/RWC/clap_embeddings/clap_embeddings_labels.csv"
    rwc_clap_embeddings_metadata = pd.read_csv(rwc_clap_embeddings_metadata_path)

    # For each sample in RWC metadata, compute the distance between its embedding and each trait embedding and save a new dataframe
    samples_traits_distances_df = rwc_clap_embeddings_metadata[rwc_clap_embeddings_metadata.columns[0:2]] # Keep "Path" and "Instrument" columns
    
    for _, row in tqdm(samples_traits_distances_df.iterrows(), total=len(samples_traits_distances_df)):
        sample_embedding = torch.load(row["Path"], weights_only=True).to(device)
        for trait_tuple in timber_traits_tuples:
            for trait in trait_tuple:
                trait_embedding = torch.load(f"data/CMTTP/timber_traits_embeddings/{trait}.pt", weights_only=True).to(device)
                distance = torch.norm(sample_embedding - trait_embedding).item()
                samples_traits_distances_df.loc[samples_traits_distances_df["Path"] == row["Path"], trait] = distance
    os.makedirs("models/cross-validation_timbre-model/CMTTP", exist_ok=True)
    samples_traits_distances_df.to_csv("models/cross-validation_timbre-model/CMTTP/samples_timber_traits_distances.csv", index=False)

    # Normalize all distances by the max distance of the dataframe
    samples_traits_distances_df[samples_traits_distances_df.columns[2:]] = samples_traits_distances_df[samples_traits_distances_df.columns[2:]].div(samples_traits_distances_df[samples_traits_distances_df.columns[2:]].max(axis=0), axis=1)
    samples_traits_distances_df.to_csv("models/cross-validation_timbre-model/CMTTP/samples_timber_traits_distances.csv", index=False)


    # for each couple of traits, keep only the min value
    samples_traits_distances_refactored_df = samples_traits_distances_df.copy()
    for trait_tuple in timber_traits_tuples:
        if len(trait_tuple) > 1:
            min_val = samples_traits_distances_refactored_df[trait_tuple].min(axis=1)
            for trait in trait_tuple:
                samples_traits_distances_refactored_df.drop(columns=[trait], inplace=True)
            samples_traits_distances_refactored_df[f"{'-'.join(trait_tuple)}"] = min_val
    samples_traits_distances_refactored_df.to_csv("models/cross-validation_timbre-model/CMTTP/samples_timber_traits_distances_refactored.csv", index=False)


    # Inverse distances to get the score
    samples_traits_inversed_distances_df = samples_traits_distances_refactored_df.copy()
    samples_traits_inversed_distances_df[samples_traits_inversed_distances_df.columns[2:]] = 1 - samples_traits_inversed_distances_df[samples_traits_inversed_distances_df.columns[2:]]
    samples_traits_inversed_distances_df.to_csv("models/cross-validation_timbre-model/CMTTP/CMTTP_predictions.csv", index=False)


    # Compute absolute errors for each trait and each sample
    samples_traits_absolute_errors_df = samples_traits_inversed_distances_df.copy()
    for index, row in tqdm(samples_traits_absolute_errors_df.iterrows(), total=len(samples_traits_absolute_errors_df)):
        if index != 0:
            continue
        instrument = row["Instrument"]
        for trait in timber_traits_names:
            predicted_value = row[trait]
            ground_truth_row = timber_traits_df.loc[timber_traits_df["RWC Name"] == instrument]
            ground_truth_value = ground_truth_row[trait].iloc[0]
            # Normalize ground truth values
            ground_truth_value = (ground_truth_value - 1) / 6.0
            samples_traits_absolute_errors_df.loc[index, trait] = abs(predicted_value - ground_truth_value)
    samples_traits_absolute_errors_df.to_csv("models/cross-validation_timbre-model/CMTTP/CMTTP_absolute_errors.csv", index=False)


    average_metric = {}
    # Group participant's data by instrument
    grouped_by_instrument = samples_traits_absolute_errors_df.groupby('Instrument')
    for instrument, instrument_df in grouped_by_instrument:
        average_metric[instrument] = {}
        to_be_averaged = []
        for trait in timber_traits_names:
            average_metric[instrument][trait] = instrument_df[trait].mean()
            to_be_averaged.append(average_metric[instrument][trait])
        average_metric[instrument]["Average"] = sum(to_be_averaged) / len(to_be_averaged)
    # Add a "Average" row
    average_metric["Average"] = {}
    for trait in timber_traits_names + ["Average"]:
        average_metric["Average"][trait] = np.mean([average_metric[instrument][trait] for instrument in average_metric.keys() if instrument != "Average"])
    
    df = pd.DataFrame.from_dict(average_metric, orient="index")
    # Place "Average" at the beginning of the dataframe
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("Average")))
    df = df[cols]
    df.to_csv("models/cross-validation_timbre-model/CMTTP/CMTTP_maes_per_instrument.csv", index=False)