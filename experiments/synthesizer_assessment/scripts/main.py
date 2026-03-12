import yaml
from train_model import train_model
from compute_mean_embeddings_RWC import compute_mean_embeddings
from create_midi_files import create_midi_files
from synthesize_samples import synthesize_all
from compute_synthesized_samples_embeddings import compute_embeddings
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    # Load config.yaml
    with open("experiments/synthesizer_assessment/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_type = config["embeddings_type"]
    hidden_layers = config["model_hidden_layers"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    epochs = config["epochs"]

    # train_model(embeddings_type, hidden_layers, learning_rate, batch_size, patience, epochs)
    # compute_mean_embeddings()
    # create_midi_files()
    synthesize_all(seed)
    ##### Compute metadata
    # compute_embeddings(embeddings_type)
    #### COmpute predictions, metrics and figures

if __name__ == "__main__":
    main()