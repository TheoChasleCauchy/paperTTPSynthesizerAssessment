import yaml
from train_model import train_model
from compute_mean_embeddings_RWC import compute_mean_embeddings
from create_midi_files import create_midi_files
from synthesize_samples import synthesize_all
from compute_synthesized_samples_metadata import compute_synth_metadata
from compute_synthesized_samples_embeddings import compute_embeddings
from compute_embeddings_metadata import compute_synthesized_samples_embeddings_metadata
from compute_predictions_metrics import compute_predictions_metrics
from TokenSynth_nearest_and_furthest_neighbors import tokenSynth_nearest_and_furthest_neighbors
from RWC_nearest_and_furthest_neighbors import RWC_nearest_and_furthest_neighbors
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
    # synthesize_all(seed)
    # compute_synth_metadata()
    # compute_embeddings(embeddings_type)
    # compute_synthesized_samples_embeddings_metadata(embeddings_type)
    # compute_predictions_metrics(embeddings_type, hidden_layers)
    # tokenSynth_nearest_and_furthest_neighbors()
    RWC_nearest_and_furthest_neighbors()

if __name__ == "__main__":
    main()