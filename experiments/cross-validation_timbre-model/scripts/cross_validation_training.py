from tqdm import tqdm
import pandas as pd
import yaml
import os

from samples_dataset import SamplesDataset
from timbre_mlp import TimbreMLP

def train_model(embeddings_type, hidden_layers, learning_rate, batch_size, patience):
    embeddings_type = embeddings_type + "_embeddings"
    
    # Load train and valid datasets metadata
    train_dataset_path = f"data/metadata/RWC/{embeddings_type}/train_{embeddings_type}_labels.csv"
    valid_dataset_path = f"data/metadata/RWC/{embeddings_type}/valid_{embeddings_type}_labels.csv"
    train_dataset_metadata = pd.read_csv(train_dataset_path)

    instruments_names = train_dataset_metadata['Instrument'].unique()
    timbre_traits_names = train_dataset_metadata.columns[2:].tolist() 

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

    model_save_folder = f"./models/cross-validation_timbre-model/timbre_model_{embeddings_type}_{hidden_layer_suffix}/"

    for excluded_instrument in tqdm(instruments_names, total=len(instruments_names), desc=f"Training models for {embeddings_type} embeddings, {len(hidden_layers)} hidden layers"):
        # Cross-Validation: For each instrument one model is trained without its samples, and we evaluate the model on these samples
        train_dataset, train_dataloader = SamplesDataset.create_dataloader(train_dataset_path, batch_size=batch_size, exclude_instrument=excluded_instrument)
        valid_dataset, valid_dataloader = SamplesDataset.create_dataloader(valid_dataset_path, batch_size=batch_size, exclude_instrument=excluded_instrument)

        model = TimbreMLP(input_size, hidden_layers, output_size)
        model_save_path = os.path.join(model_save_folder, f"timbre_model_{embeddings_type}_{hidden_layer_suffix}_{excluded_instrument.replace(' ', '_')}")
        model.train_model(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, learning_rate=learning_rate, patience=patience, model_save_path=model_save_path)


def train_all_models():
    # Load config.yaml
    with open("experiments/timbre-model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_type = config["embeddings_type"]
    model_hidden_layers = config["model_hidden_layers"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]

    for emb_type in embeddings_type:
        for hidden_layers_conf in model_hidden_layers:
            train_model(emb_type, hidden_layers_conf, learning_rate, batch_size, patience)