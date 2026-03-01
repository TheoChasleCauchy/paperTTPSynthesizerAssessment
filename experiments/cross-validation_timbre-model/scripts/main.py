from cross_validation_training import train_all_models
from compute_predictions_metrics import compute_predictions_metrics
from timber_traits_predictions_radar_charts import plot_all_instruments_radar_charts


def main():
    train_all_models()
    compute_predictions_metrics()
    plot_all_instruments_radar_charts()


if __name__ == "__main__":
    main()