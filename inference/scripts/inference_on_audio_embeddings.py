import os
import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np
from timbre_mlp import TimbreMLP

def inference(audios_folder: str, model_save_folder: str):
    # Get the model architecture from the checkpoint
    with open(os.path.join(model_save_folder, "model_architecture.yaml"), "r") as f:
        model_architecture = yaml.safe_load(f)

    input_size = model_architecture["input_size"]
    hidden_layers = model_architecture["hidden_layers"]
    output_size = model_architecture["output_size"]

    # Load the dataset metadata
    timber_traits_path = "data/Reymore/timber_traits_ground_truth.csv"
    timber_traits_df = pd.read_csv(timber_traits_path)
    timbre_traits_names = timber_traits_df.columns[2:].tolist()

    dataset_path = os.path.join("inference", "results", os.path.basename(audios_folder), "inference_results.csv")
    audios_df = pd.read_csv(dataset_path)

    # Load the pre-trained model
    model = TimbreMLP.load_model(
        f"{model_save_folder}/timbre_mlp.pth",
        input_size=input_size,
        hidden_sizes=hidden_layers,
        output_size=output_size,
    )

    predictions = []

    # Compute predictions for each sample
    for row in tqdm(audios_df.itertuples(index=False), total=len(audios_df), desc="Computing predictions for all samples"):

        embedding_path = row.embedding_path

        # Evaluate the model and get predictions
        predicted_values = model.only_one_embedding_blind_evaluation(embedding_path).cpu().detach().numpy()
        predictions.append(predicted_values)
    
    # Iterate over each row and add the new columns
    for timbre_trait_id, timbre_trait in enumerate(timbre_traits_names):
        # Add a new column for each trait
        audios_df[timbre_trait] = [predictions[i][timbre_trait_id] for i in range(len(audios_df))]

    # Save the updated DataFrame to a new CSV file
    audios_df.to_csv(os.path.join("inference", "results", os.path.basename(audios_folder), "inference_results.csv"), index=False)