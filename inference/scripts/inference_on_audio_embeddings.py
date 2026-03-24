import os  # Operating system interfaces for directory and file operations
import yaml  # YAML file handling
import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bar for iterative tasks
import numpy as np  # Numerical operations
from timbre_mlp import TimbreMLP  # Custom MLP model for timbre prediction

def inference(audios_folder: str, model_save_folder: str):
    """
    Perform inference on audio samples using a pre-trained TimbreMLP model.

    Args:
        audios_folder (str): Path to the folder containing audio samples and their embeddings.
        model_save_folder (str): Path to the folder containing the pre-trained model and its architecture configuration.

    This function:
    1. Loads the model architecture from a YAML file.
    2. Loads the dataset metadata and timber trait names.
    3. Loads the pre-trained model.
    4. Computes predictions for each audio sample.
    5. Saves the predictions to the dataset metadata CSV file.

    Steps:
    - Load the model architecture from a YAML file.
    - Load the dataset metadata and timber trait names.
    - Load the pre-trained model.
    - For each audio sample, compute predictions using the model.
    - Save the predictions to the dataset metadata CSV file.

    Returns:
        None: Predictions are saved to the dataset metadata CSV file.
    """
    # Load the model architecture from the YAML file
    with open(os.path.join(model_save_folder, "model_architecture.yaml"), "r") as f:
        model_architecture = yaml.safe_load(f)

    # Extract model architecture parameters
    input_size = model_architecture["input_size"]
    hidden_layers = model_architecture["hidden_layers"]
    output_size = model_architecture["output_size"]

    # Load timber trait names from the ground truth CSV file
    timber_traits_path = "data/Reymore/timber_traits_ground_truth.csv"
    timber_traits_df = pd.read_csv(timber_traits_path)
    timbre_traits_names = timber_traits_df.columns[2:].tolist()

    # Load the dataset metadata for the audio samples
    dataset_path = os.path.join("inference", "results", os.path.basename(audios_folder), "inference_results.csv")
    audios_df = pd.read_csv(dataset_path)

    # Load the pre-trained model
    model = TimbreMLP.load_model(
        f"{model_save_folder}/timbre_mlp.pth",
        input_size=input_size,
        hidden_sizes=hidden_layers,
        output_size=output_size,
    )

    # Initialize a list to store predictions
    predictions = []

    # Compute predictions for each audio sample
    for row in tqdm(audios_df.itertuples(index=False), total=len(audios_df), desc="Computing predictions for all samples"):
        # Get the path to the embedding
        embedding_path = row.embedding_path

        # Evaluate the model and get predictions for the embedding
        predicted_values = model.only_one_embedding_blind_evaluation(embedding_path).cpu().detach().numpy()
        predictions.append(predicted_values)

    # Add predictions as new columns to the dataset metadata
    for timbre_trait_id, timbre_trait in enumerate(timbre_traits_names):
        # Add a new column for each timber trait
        audios_df[timbre_trait] = [predictions[i][timbre_trait_id] for i in range(len(audios_df))]

    # Save the updated dataset metadata to a CSV file
    audios_df.to_csv(os.path.join("inference", "results", os.path.basename(audios_folder), "inference_results.csv"), index=False)
