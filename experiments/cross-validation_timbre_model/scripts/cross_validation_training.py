from tqdm import tqdm  # Progress bar for iterative tasks
import pandas as pd  # Data manipulation and analysis
import yaml  # YAML file handling
import os  # Operating system interfaces for directory and file operations
from samples_dataset import SamplesDataset  # Custom dataset class for handling samples
from timbre_mlp import TimbreMLP  # Custom MLP model for timbre prediction

def train_model(embeddings_type, hidden_layers, learning_rate, batch_size, patience, epochs):
    """
    Train a TimbreMLP model using cross-validation, excluding one instrument at a time.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        hidden_layers (list): List of hidden layer sizes for the TimbreMLP model.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for the DataLoader.
        patience (int): Number of epochs to wait before early stopping.
        epochs (int): Number of training epochs.

    This function:
    1. Loads the train and validation datasets metadata.
    2. For each instrument, trains a model excluding that instrument's samples.
    3. Saves the trained model for each excluded instrument.

    Steps:
    - Load the train and validation datasets metadata.
    - For each instrument, create DataLoaders excluding that instrument's samples.
    - Train a TimbreMLP model and save it to disk.

    Returns:
        None: Trained models are saved to disk.
    """
    # Append "_embeddings" to the embeddings type
    embeddings_type = embeddings_type + "_embeddings"

    # Load train dataset metadata
    train_dataset_path = f"data/metadata/RWC/{embeddings_type}/train_{embeddings_type}_labels.csv"
    valid_dataset_path = f"data/metadata/RWC/{embeddings_type}/valid_{embeddings_type}_labels.csv"
    train_dataset_metadata = pd.read_csv(train_dataset_path)

    # Get unique instrument names
    instruments_names = train_dataset_metadata['Instrument'].unique()

    # Determine the input size based on the embedding type
    match embeddings_type:
        case "clap_embeddings":
            input_size = 512
        case "clap-music_embeddings":
            input_size = 512
        case "vggish_embeddings":
            input_size = 128
        case "mert_embeddings":
            input_size = 768

    # Determine the hidden layer suffix based on the number of hidden layers
    match len(hidden_layers):
        case 0:
            hidden_layer_suffix = "no_hidden_layers"
        case 1:
            hidden_layer_suffix = "single_hidden_layer"
        case _:
            hidden_layer_suffix = f"{len(hidden_layers)}_hidden_layers"

    # Set the output size to 20 (number of timber traits)
    output_size = 20

    # Create the model save folder
    model_save_folder = f"models/cross-validation_timbre_model/timbre_model_{embeddings_type}_{hidden_layer_suffix}/"

    # Train a model for each excluded instrument
    for excluded_instrument in tqdm(instruments_names, total=len(instruments_names), desc=f"Training models for {embeddings_type} embeddings, {len(hidden_layers)} hidden layers"):
        # Cross-Validation: For each instrument, train a model without its samples
        _, train_dataloader = SamplesDataset.create_dataloader(train_dataset_path, batch_size=batch_size, exclude_instrument=excluded_instrument, shuffle=True)
        _, valid_dataloader = SamplesDataset.create_dataloader(valid_dataset_path, batch_size=batch_size, exclude_instrument=excluded_instrument, shuffle=False)

        # Create the model save path
        model_save_path = os.path.join(model_save_folder, f"timbre_model_{embeddings_type}_{hidden_layer_suffix}_{excluded_instrument.replace(' ', '_')}")

        # Initialize and train the model
        model = TimbreMLP(input_size, hidden_layers, output_size, save_path=model_save_path)
        model.train_model(train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, learning_rate=learning_rate, patience=patience, epochs=epochs)

def train_all_models():
    """
    Train all TimbreMLP models for different embedding types and hidden layer configurations.

    This function:
    1. Loads the configuration from a YAML file.
    2. For each embedding type and hidden layer configuration, trains the models.

    Steps:
    - Load the configuration from a YAML file.
    - For each embedding type and hidden layer configuration, call the `train_model` function.

    Returns:
        None: All trained models are saved to disk.
    """
    # Load the configuration from the YAML file
    with open("experiments/cross-validation_timbre_model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    embeddings_types = config["embeddings_types"]
    model_hidden_layers = config["model_hidden_layers"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    epochs = config["epochs"]

    # Train models for each embedding type and hidden layer configuration
    for emb_type in embeddings_types:
        for hidden_layers_conf in model_hidden_layers:
            train_model(emb_type, hidden_layers_conf, learning_rate, batch_size, patience, epochs)
