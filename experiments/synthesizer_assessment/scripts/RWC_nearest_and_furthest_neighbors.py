import pandas as pd  # Data manipulation and analysis
import torch  # PyTorch for tensor operations
import os  # Operating system interfaces for directory and file operations
import librosa  # Audio and music analysis library
import soundfile as sf  # Library for reading and writing sound files
from tqdm import tqdm  # Progress bar for iterative tasks
from wav_to_spectrogram import wav_to_spectrogram  # Custom function to convert WAV to spectrogram

def compute_nearest_neighbors():
    """
    Compute and save the nearest neighbor audio samples for each instrument in the RWC dataset.

    This function:
    1. Loads the predictions and ground truth data for RWC samples.
    2. For each instrument, finds the RWC sample closest to the ground truth.
    3. Saves the nearest neighbor audio and its spectrogram.

    Steps:
    - Load the predictions and ground truth data.
    - For each instrument, compute the Euclidean distance between each RWC sample and the ground truth.
    - Save the nearest neighbor audio and its spectrogram to disk.

    Returns:
        None: Nearest neighbor audio files and spectrograms are saved to disk.
    """
    # Load predictions metadata
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # Load ground truth data
    ground_truth_path = "data/Reymore/timbre_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timbre_traits_column = ground_truth_df.columns[2:]  # Get all label columns (excluding "RWC Name" and "Instrument")

    # Get unique instrument names
    instrument_names = metadata_df["Instrument"].unique()

    # Create output directory
    output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/nearest_neighbors_over_all_traits"
    os.makedirs(output_folder, exist_ok=True)

    for instrument in tqdm(instrument_names, desc=f"Processing nearest neighbors"):
        # Filter metadata for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get ground truth labels for the current instrument
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timbre_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        # Normalize ground truth labels
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get embedding paths for the current instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Initialize variables to track the nearest neighbor
        min_distance = float("inf")
        nearest_neighbor_index = -1

        # Find the nearest neighbor
        for i, row in enumerate(instrument_df.itertuples(index=False)):
            sample_predictions = row[2:]
            sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)
            distance = torch.norm(sample_predictions - ground_truth_labels_vector).item()

            if distance < min_distance:
                nearest_neighbor_index = i
                min_distance = distance

        # Get the basename of the nearest neighbor
        basename = os.path.basename(embedding_paths.iloc[nearest_neighbor_index])
        underscore_index = basename.find("_")
        if underscore_index != -1:
            basename = basename[underscore_index + 1:]
        basename = basename.replace("_embedding.pt", ".wav")

        # Load and save the nearest neighbor audio
        audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
        audio = librosa.load(audio_path)
        output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_nearest_neighbor_dist_{min_distance:.2f}_{basename}")
        sf.write(output_audio_path, audio[0], samplerate=audio[1])
        wav_to_spectrogram(output_audio_path)

def compute_nearest_neighbor_each_trait():
    """
    Compute and save the nearest neighbor audio samples for each instrument and timbre trait in the RWC dataset.

    This function:
    1. Loads the predictions and ground truth data for RWC samples.
    2. For each instrument and timbre trait, finds the RWC sample closest to the ground truth.
    3. Saves the nearest neighbor audio and its spectrogram.

    Steps:
    - Load the predictions and ground truth data.
    - For each instrument and timbre trait, compute the Euclidean distance between each RWC sample and the ground truth.
    - Save the nearest neighbor audio and its spectrogram to disk.

    Returns:
        None: Nearest neighbor audio files and spectrograms are saved to disk.
    """
    # Load predictions metadata
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # Load ground truth data
    ground_truth_path = "data/Reymore/timbre_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timbre_traits_column = ground_truth_df.columns[2:]

    # Get unique instrument names
    instrument_names = metadata_df["Instrument"].unique()

    for instrument in tqdm(instrument_names, desc=f"Processing nearest neighbors (each trait)"):
        # Filter metadata for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get ground truth labels for the current instrument
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timbre_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get embedding paths for the current instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Create output directory
        output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/nearest_neighbors_each_trait/{instrument.replace(' ', '_')}"
        os.makedirs(output_folder, exist_ok=True)

        # Initialize variables to track the nearest neighbor for each trait
        min_distance = {timbre_trait: float("inf") for timbre_trait in timbre_traits_column}
        nearest_neighbor_index = {timbre_trait: -1 for timbre_trait in timbre_traits_column}

        # Find the nearest neighbor for each trait
        for i, row in enumerate(instrument_df.itertuples(index=False)):
            for timbre_trait_ind, timbre_trait in enumerate(timbre_traits_column):
                sample_predictions = row[timbre_trait_ind + 2]
                sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)
                distance = torch.norm(sample_predictions - ground_truth_labels_vector[0, timbre_trait_ind]).item()

                if distance < min_distance[timbre_trait]:
                    nearest_neighbor_index[timbre_trait] = i
                    min_distance[timbre_trait] = distance

        # Save the nearest neighbor audio for each trait
        for timbre_trait in timbre_traits_column:
            basename = os.path.basename(embedding_paths.iloc[nearest_neighbor_index[timbre_trait]])
            underscore_index = basename.find("_")
            if underscore_index != -1:
                basename = basename[underscore_index + 1:]
            basename = basename.replace("_embedding.pt", ".wav")

            audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
            audio = librosa.load(audio_path)
            output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_nearest_neighbor_trait_{timbre_trait}_dist_{min_distance[timbre_trait]:.2f}_{basename}")
            sf.write(output_audio_path, audio[0], samplerate=audio[1])
            wav_to_spectrogram(output_audio_path)

def compute_furthest_neighbors():
    """
    Compute and save the furthest neighbor audio samples for each instrument in the RWC dataset.

    This function:
    1. Loads the predictions and ground truth data for RWC samples.
    2. For each instrument, finds the RWC sample furthest from the ground truth.
    3. Saves the furthest neighbor audio and its spectrogram.

    Steps:
    - Load the predictions and ground truth data.
    - For each instrument, compute the Euclidean distance between each RWC sample and the ground truth.
    - Save the furthest neighbor audio and its spectrogram to disk.

    Returns:
        None: Furthest neighbor audio files and spectrograms are saved to disk.
    """
    # Load predictions metadata
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # Load ground truth data
    ground_truth_path = "data/Reymore/timbre_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timbre_traits_column = ground_truth_df.columns[2:]

    # Get unique instrument names
    instrument_names = metadata_df["Instrument"].unique()

    # Create output directory
    output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/furthest_neighbors_over_all_traits"
    os.makedirs(output_folder, exist_ok=True)

    for instrument in tqdm(instrument_names, desc=f"Processing furthest neighbors"):
        # Filter metadata for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get ground truth labels for the current instrument
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timbre_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get embedding paths for the current instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Initialize variables to track the furthest neighbor
        max_distance = float("-inf")
        furthest_neighbor_index = -1

        # Find the furthest neighbor
        for i, row in enumerate(instrument_df.itertuples(index=False)):
            sample_predictions = row[2:]
            sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)
            distance = torch.norm(sample_predictions - ground_truth_labels_vector).item()

            if distance > max_distance:
                furthest_neighbor_index = i
                max_distance = distance

        # Get the basename of the furthest neighbor
        basename = os.path.basename(embedding_paths.iloc[furthest_neighbor_index])
        underscore_index = basename.find("_")
        if underscore_index != -1:
            basename = basename[underscore_index + 1:]
        basename = basename.replace("_embedding.pt", ".wav")

        # Load and save the furthest neighbor audio
        audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
        audio = librosa.load(audio_path)
        output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_furthest_neighbor_dist_{max_distance:.2f}_{basename}")
        sf.write(output_audio_path, audio[0], samplerate=audio[1])
        wav_to_spectrogram(output_audio_path)

def compute_furthest_neighbor_each_trait():
    """
    Compute and save the furthest neighbor audio samples for each instrument and timbre trait in the RWC dataset.

    This function:
    1. Loads the predictions and ground truth data for RWC samples.
    2. For each instrument and timbre trait, finds the RWC sample furthest from the ground truth.
    3. Saves the furthest neighbor audio and its spectrogram.

    Steps:
    - Load the predictions and ground truth data.
    - For each instrument and timbre trait, compute the Euclidean distance between each RWC sample and the ground truth.
    - Save the furthest neighbor audio and its spectrogram to disk.

    Returns:
        None: Furthest neighbor audio files and spectrograms are saved to disk.
    """
    # Load predictions metadata
    metadata_path = f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv"
    metadata_df = pd.read_csv(metadata_path)

    # Load ground truth data
    ground_truth_path = "data/Reymore/timbre_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timbre_traits_column = ground_truth_df.columns[2:]

    # Get unique instrument names
    instrument_names = metadata_df["Instrument"].unique()

    for instrument in tqdm(instrument_names, desc=f"Processing furthest neighbors (each trait)"):
        # Filter metadata for the current instrument
        instrument_df = metadata_df[metadata_df["Instrument"] == instrument]

        # Get ground truth labels for the current instrument
        ground_truth_labels = ground_truth_df[ground_truth_df["RWC Name"] == instrument][timbre_traits_column]
        ground_truth_labels_vector = torch.from_numpy(ground_truth_labels.values).float()
        ground_truth_labels_vector = (ground_truth_labels_vector - 1) / 6.0

        # Get embedding paths for the current instrument
        embedding_paths_column = instrument_df.columns[0]
        embedding_paths = instrument_df[embedding_paths_column]

        # Create output directory
        output_folder = f"experiments/synthesizer_assessment/results/nearest_and_furthest_neighbors/RWC/furthest_neighbors_each_trait/{instrument.replace(' ', '_')}"
        os.makedirs(output_folder, exist_ok=True)

        # Initialize variables to track the furthest neighbor for each trait
        max_distance = {timbre_trait: float("-inf") for timbre_trait in timbre_traits_column}
        furthest_neighbor_index = {timbre_trait: -1 for timbre_trait in timbre_traits_column}

        # Find the furthest neighbor for each trait
        for i, row in enumerate(instrument_df.itertuples(index=False)):
            for timbre_trait_ind, timbre_trait in enumerate(timbre_traits_column):
                sample_predictions = row[timbre_trait_ind + 2]
                sample_predictions = torch.tensor(sample_predictions, dtype=torch.float32)
                distance = torch.norm(sample_predictions - ground_truth_labels_vector[0, timbre_trait_ind]).item()

                if distance > max_distance[timbre_trait]:
                    furthest_neighbor_index[timbre_trait] = i
                    max_distance[timbre_trait] = distance

        # Save the furthest neighbor audio for each trait
        for timbre_trait in timbre_traits_column:
            basename = os.path.basename(embedding_paths.iloc[furthest_neighbor_index[timbre_trait]])
            underscore_index = basename.find("_")
            if underscore_index != -1:
                basename = basename[underscore_index + 1:]
            basename = basename.replace("_embedding.pt", ".wav")

            audio_path = os.path.join("data/RWC/RWC-preprocessed/", instrument, basename)
            audio = librosa.load(audio_path)
            output_audio_path = os.path.join(output_folder, f"{instrument}_ground_truth_furthest_neighbor_trait_{timbre_trait}_dist_{max_distance[timbre_trait]:.2f}_{basename}")
            sf.write(output_audio_path, audio[0], samplerate=audio[1])
            wav_to_spectrogram(output_audio_path)

def RWC_nearest_and_furthest_neighbors():
    """
    Compute and save the nearest and furthest neighbor audio samples for all instruments in the RWC dataset.

    This function calls the individual functions to compute and save the nearest and furthest neighbors.

    Steps:
    - Call functions to compute and save nearest and furthest neighbors for all instruments.

    Returns:
        None: All results are saved to disk.
    """
    compute_nearest_neighbors()
    compute_furthest_neighbors()
    compute_nearest_neighbor_each_trait()
    compute_furthest_neighbor_each_trait()
