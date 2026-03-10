from cross_validation_training import train_all_models
from compute_predictions_metrics import compute_predictions_metrics
from timber_traits_predictions_radar_charts import plot_all_instruments_radar_charts
from CMTTP import CMTTP
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    CMTTP()
    train_all_models()
    compute_predictions_metrics()
    plot_all_instruments_radar_charts()


if __name__ == "__main__":
    main()