from train_model import train_model
from compute_mean_embeddings_RWC import compute_mean_embeddings
from create_midi_files import create_midi_files
from synthesize_samples import synthesize_all
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    train_model()
    compute_mean_embeddings()
    create_midi_files()
    synthesize_all(seed)

if __name__ == "__main__":
    main()