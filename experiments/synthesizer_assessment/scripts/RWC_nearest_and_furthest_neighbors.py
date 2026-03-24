import pandas as pd
import torch
import os
import librosa
import soundfile as sf
from tqdm import tqdm
from wav_to_spectrogram import wav_to_spectrogram

def compute_nearest_neighbors():
    # 1. Load embeddings predictions
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # 2. Load ground truth
    ground_truth_path = "data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timber_traits_column = ground_truth_df.columns[2:]  # Get all label columns (excluding "RWC Name" and "Instrument")

    # 3. Find nearest neighbor from the ground truth in each instrument
    instrument_names = metadata_df["Instrument"].unique()

    output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/nearest_neighbors_over_all_traits"
    os.makedirs(output_folder, exist_ok=True)

    for instrument in tqdm(instrument_names, desc=f"Processing nearest neighbors"):
        # Select rows for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get the ground truth labels for this instrument (columns [2:])
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timber_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        # Normalize labels
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get the embedding paths for this instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Compute the nearest neighbor for each sample in the instrument
        min_distance = float("inf")
        nearest_neighbor_index = -1

        for i, row in enumerate(instrument_df.itertuples(index=False)):
            # Load the predicted values
            sample_predictions = row[2:]

            # Convert to tensor
            sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)

            # Compute distance to ground truth
            distance = torch.norm(sample_predictions - ground_truth_labels_vector).item()
            
            if distance < min_distance:
                # Find the index of the nearest neighbor
                nearest_neighbor_index = i
                min_distance = distance

        # Get the basename of the nearest neighbor
        basename = os.path.basename(embedding_paths.iloc[nearest_neighbor_index])

        # Remove the term before the first "_" and the "_"
        underscore_index = basename.find("_")
        if underscore_index != -1:
            basename = basename[underscore_index + 1:]
        
        # Replace "_embedding.pt" with ".wav"
        basename = basename.replace("_embedding.pt", ".wav")

        # Get the corresponding audio
        audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
        audio = librosa.load(audio_path)
            
        # Save the path of the nearest neighbor
        output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_nearest_neighbor_dist_{min_distance:.2f}_{basename}")
        sf.write(output_audio_path, audio[0], samplerate=audio[1])
        # print(f"Saved nearest neighbor for instrument {instrument} to {output_audio_path}")
        wav_to_spectrogram(output_audio_path)

def compute_nearest_neighbor_each_trait():
    # 1. Load embeddings predictions
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # 2. Load ground truth
    ground_truth_path = "data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timber_traits_column = ground_truth_df.columns[2:]  # Get all label columns (excluding "RWC Name" and "Instrument")

    # 3. Find nearest neighbor from the ground truth in each instrument
    instrument_names = metadata_df["Instrument"].unique()

    for instrument in tqdm(instrument_names, desc=f"Processing nearest neighbors (each trait)"):
        # Select rows for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get the ground truth labels for this instrument (columns [2:])
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timber_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        # Normalize labels
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get the embedding paths for this instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Compute the nearest neighbor for each sample in the instrument
        output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/nearest_neighbors_each_trait/{instrument.replace(' ', '_')}"
        os.makedirs(output_folder, exist_ok=True)

        min_distance = {timber_trait: float("inf") for timber_trait in timber_traits_column}
        nearest_neighbor_index = {timber_trait: -1 for timber_trait in timber_traits_column}

        for i, row in enumerate(instrument_df.itertuples(index=False)):
            for timber_trait_ind, timber_trait in enumerate(timber_traits_column):
                # Load the predicted values
                sample_predictions = row[timber_trait_ind + 2]  # +2 because first two columns are "Instrument" and "Embedding Path"

                # Convert to tensor
                sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)

                # Compute distance to ground truth
                distance = torch.norm(sample_predictions - ground_truth_labels_vector[0,timber_trait_ind]).item()
                
                if distance < min_distance[timber_trait]:
                    # Find the index of the nearest neighbor
                    nearest_neighbor_index[timber_trait] = i
                    min_distance[timber_trait] = distance

        for timber_trait in timber_traits_column:
            # Get the basename of the nearest neighbor
            basename = os.path.basename(embedding_paths.iloc[nearest_neighbor_index[timber_trait]])

            # Remove the term before the first "_" and the "_"
            underscore_index = basename.find("_")
            if underscore_index != -1:
                basename = basename[underscore_index + 1:]
            
            # Replace "_embedding.pt" with ".wav"
            basename = basename.replace("_embedding.pt", ".wav")

            # Get the corresponding audio
            audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
            audio = librosa.load(audio_path)
                
            # Save the path of the nearest neighbor
            output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_nearest_neighbor_trait_{timber_trait}_dist_{min_distance[timber_trait]:.2f}_{basename}")
            sf.write(output_audio_path, audio[0], samplerate=audio[1])
            wav_to_spectrogram(output_audio_path)

def compute_furthest_neighbors():
    # 1. Load embeddings predictions
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # 2. Load ground truth
    ground_truth_path = "data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timber_traits_column = ground_truth_df.columns[2:]  # Get all label columns (excluding "RWC Name" and "Instrument")

    # 3. Find nearest neighbor from the ground truth in each instrument
    instrument_names = metadata_df["Instrument"].unique()

    output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/furthest_neighbors_over_all_traits"
    os.makedirs(output_folder, exist_ok=True)

    for instrument in tqdm(instrument_names, desc=f"Processing furthest neighbors"):
        # Select rows for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get the ground truth labels for this instrument (columns [2:])
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timber_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        # Normalize labels
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get the embedding paths for this instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        max_distance = float("-inf")
        furthest_neighbor_index = -1

        for i, row in enumerate(instrument_df.itertuples(index=False)):
            # Load the predicted values
            sample_predictions = row[2:]

            # Convert to tensor
            sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)

            # Compute distance to ground truth
            distance = torch.norm(sample_predictions - ground_truth_labels_vector).item()
            
            if distance > max_distance:
                # Find the index of the furthest neighbor
                furthest_neighbor_index = i
                max_distance = distance

        # Get the basename of the furthest neighbor
        basename = os.path.basename(embedding_paths.iloc[furthest_neighbor_index])

        # Remove the term before the first "_" and the "_"
        underscore_index = basename.find("_")
        if underscore_index != -1:
            basename = basename[underscore_index + 1:]
        
        # Replace "_embedding.pt" with ".wav"
        basename = basename.replace("_embedding.pt", ".wav")

        # Get the corresponding audio
        audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
        audio = librosa.load(audio_path)
            
        # Save the path of the furthest neighbor
        output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_furthest_neighbor_dist_{max_distance:.2f}_{basename}")
        sf.write(output_audio_path, audio[0], samplerate=audio[1])
        # print(f"Saved furthest neighbor for instrument {instrument} to {output_audio_path}")
        wav_to_spectrogram(output_audio_path)

def compute_furthest_neighbor_each_trait():
    # 1. Load embeddings predictions
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # 2. Load ground truth
    ground_truth_path = "data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timber_traits_column = ground_truth_df.columns[2:]  # Get all label columns (excluding "RWC Name" and "Instrument")

    # 3. Find furthest neighbor from the ground truth in each instrument
    instrument_names = metadata_df["Instrument"].unique()

    for instrument in tqdm(instrument_names, desc=f"Processing furthest neighbors (each trait)"):
        # Select rows for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get the ground truth labels for this instrument (columns [2:])
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timber_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        # Normalize labels
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get the embedding paths for this instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Compute the furthest neighbor for each sample in the instrument
        output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/furthest_neighbors_each_trait/{instrument.replace(' ', '_')}"
        os.makedirs(output_folder, exist_ok=True)

        max_distance = {timber_trait: float("-inf") for timber_trait in timber_traits_column}
        furthest_neighbor_index = {timber_trait: -1 for timber_trait in timber_traits_column}

        for i, row in enumerate(instrument_df.itertuples(index=False)):
            for timber_trait_ind, timber_trait in enumerate(timber_traits_column):
                # Load the predicted values
                sample_predictions = row[timber_trait_ind + 2]  # +2 because first two columns are "Instrument" and "Embedding Path"

                # Convert to tensor
                sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)

                # Compute distance to ground truth
                distance = torch.norm(sample_predictions - ground_truth_labels_vector[0,timber_trait_ind]).item()
                
                if distance > max_distance[timber_trait]:
                    # Find the index of the furthest neighbor
                    furthest_neighbor_index[timber_trait] = i
                    max_distance[timber_trait] = distance

        for timber_trait in timber_traits_column:
            # Get the basename of the furthest neighbor
            basename = os.path.basename(embedding_paths.iloc[furthest_neighbor_index[timber_trait]])

            # Remove the term before the first "_" and the "_"
            underscore_index = basename.find("_")
            if underscore_index != -1:
                basename = basename[underscore_index + 1:]
            
            # Replace "_embedding.pt" with ".wav"
            basename = basename.replace("_embedding.pt", ".wav")

            # Get the corresponding audio
            audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
            audio = librosa.load(audio_path)
                
            # Save the path of the furthest neighbor
            output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_furthest_neighbor_trait_{timber_trait}_dist_{max_distance[timber_trait]:.2f}_{basename}")
            sf.write(output_audio_path, audio[0], samplerate=audio[1])
            wav_to_spectrogram(output_audio_path)

def RWC_nearest_and_furthest_neighbors():
    compute_nearest_neighbors()
    compute_furthest_neighbors()
    compute_nearest_neighbor_each_trait()
    compute_furthest_neighbor_each_trait()