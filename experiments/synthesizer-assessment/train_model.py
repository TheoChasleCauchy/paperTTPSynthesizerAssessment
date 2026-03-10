from tqdm import tqdm
import pandas as pd
import yaml
import os
from samples_dataset import SamplesDataset
from timbre_mlp import TimbreMLP

def train_model(embeddings_type, learning_rate, batch_size, patience, epochs):
    # Load config.yaml
    with open("experiments/synthesizer-assessment/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_type = config["embeddings_type"]
    hidden_layers = config["model_hidden_layers"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    epochs = config["epochs"]

    embeddings_type = embeddings_type + "_embeddings"
    
    # Load train and valid datasets metadata
    train_dataset_path = f"data/metadata/RWC/{embeddings_type}/train_{embeddings_type}_labels.csv"
    valid_dataset_path = f"data/metadata/RWC/{embeddings_type}/valid_{embeddings_type}_labels.csv"

    # Model structure
    match embeddings_type:
        case "clap_embeddings":
            input_size = 512
        case "clap-music_embeddings":
            input_size = 512
        case "vggish_embeddings":
            input_size = 128
        case "mert_embeddings":
            input_size = 768

    match len(hidden_layers):
        case 0:
            hidden_layer_suffix = "no_hidden_layers"
        case 1:
            hidden_layer_suffix = f"single_hidden_layer"
        case _:
            hidden_layer_suffix = f"{len(hidden_layers)}_hidden_layers"

    output_size = 20 # 20 timber traits

    model_save_folder = f"models/synthesizer-assessment/timbre_model_{embeddings_type}_{hidden_layer_suffix}/"

    # Cross-Validation: For each instrument one model is trained without its samples, and we evaluate the model on these samples
    _, train_dataloader = SamplesDataset.create_dataloader(train_dataset_path, batch_size=batch_size, shuffle=True)
    _, valid_dataloader = SamplesDataset.create_dataloader(valid_dataset_path, batch_size=batch_size, shuffle=False)

    model_save_path = os.path.join(model_save_folder, f"timbre_model_{embeddings_type}_{hidden_layer_suffix}")
    model = TimbreMLP(input_size, hidden_layers, output_size, save_path=model_save_path)
    model.train_model(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, learning_rate=learning_rate, patience=patience, epochs=epochs)

