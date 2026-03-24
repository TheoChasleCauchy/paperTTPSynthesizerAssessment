import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces for directory and file operations
from timbre_mlp import TimbreMLP  # Custom MLP model for timbre prediction
from samples_dataset import SamplesDataset  # Custom dataset class for handling samples
from scipy.stats import pearsonr  # Pearson correlation coefficient
from tqdm import tqdm  # Progress bar for iterative tasks
import yaml  # YAML file handling

def compute_predictions(embeddings_type, hidden_layers_conf, hidden_layers_suffix):
    """
    Compute predictions for timber traits using a pre-trained TimbreMLP model.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        hidden_layers_conf (list): List of hidden layer sizes for the TimbreMLP model.
        hidden_layers_suffix (str): Suffix for the model save path based on hidden layers.

    This function:
    1. Loads the dataset metadata.
    2. For each sample, loads the pre-trained model and computes predictions.
    3. Saves the predictions to a CSV file.

    Steps:
    - Load the dataset metadata.
    - For each sample, load the pre-trained model and compute predictions.
    - Save the predictions to a CSV file.

    Returns:
        None: Predictions are saved to a CSV file.
    """
    print(f"Computing Timber Traits Predictions with the models trained on {embeddings_type} with {hidden_layers_suffix}")

    # Set the output size to 20 (number of timber traits)
    output_size = 20

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
        case _:
            raise ValueError(f"Unsupported embedding type: {embeddings_type}")

    # Create the model save folder path
    model_save_folder = f"./models/cross-validation_timbre_model/timbre_model_{embeddings_type}_{hidden_layers_suffix}/"

    # Load the dataset metadata
    dataset_path = f"data/metadata/RWC/{embeddings_type}/{embeddings_type}_labels.csv"
    dataset = pd.read_csv(dataset_path)
    timbre_traits_names = dataset.columns[2:].tolist()
    df = []

    # Compute predictions for each sample
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc="Computing predictions for all samples"):
        instrument = row.Instrument
        model_save_path = os.path.join(model_save_folder, f"timbre_model_{embeddings_type}_{hidden_layers_suffix}_{instrument.replace(' ', '_')}")

        # Load the pre-trained model
        model = TimbreMLP.load_model(
            f"{model_save_path}/timbre_mlp.pth",
            input_size=input_size,
            hidden_sizes=hidden_layers_conf,
            output_size=output_size,
        )

        # Create a DataLoader for the current sample
        evalDataset, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

        # Evaluate the model and get predictions
        eval_loss, predicted_values, _ = model.evaluate_model(evalDataloader)

        # Append predictions to the DataFrame
        df.append({
            "Sample": row.Path,
            "Excluded Instrument": instrument,
            **{timber_trait: predicted_values[:, timber_trait_id].item() for timber_trait_id, timber_trait in enumerate(timbre_traits_names)}
        })

    # Save the predictions to a CSV file
    save_path = f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_predictions.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(df)
    df.to_csv(save_path, index=False)

def compute_errors(embeddings_type, hidden_layers_suffix):
    """
    Compute absolute errors between predicted and ground truth timber traits.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        hidden_layers_suffix (str): Suffix for the model save path based on hidden layers.

    This function:
    1. Loads the predictions and ground truth data.
    2. For each sample, computes the absolute difference between predicted and ground truth values.
    3. Saves the absolute errors to a CSV file.

    Steps:
    - Load the predictions and ground truth data.
    - For each sample, compute the absolute difference between predicted and ground truth values.
    - Save the absolute errors to a CSV file.

    Returns:
        None: Absolute errors are saved to a CSV file.
    """
    print(f"Computing absolute errors for the models trained on {embeddings_type} with hidden layers {hidden_layers_suffix}")

    # Load the predictions
    predictions_path = f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_predictions.csv"
    predictions_df = pd.read_csv(predictions_path)

    # Load the ground truth data
    ground_truth_path = f"data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)

    # Create a mapping from RWC Name to ground truth values
    ground_truth_mapping = {}
    for _, row in ground_truth_df.iterrows():
        ground_truth_mapping[row["RWC Name"]] = row[2:]  # Skip first two columns (Instrument, RWC Name)

    # Compute absolute errors for each sample
    for _, row in predictions_df.iterrows():
        instrument = row["Excluded Instrument"]
        if instrument in ground_truth_mapping:
            ground_truth_values = ground_truth_mapping[instrument]
            normalized_ground_truth_values = (ground_truth_values - 1) / 6.0  # Normalize to [0,1]
            for i, timber_trait in enumerate(ground_truth_df.columns[2:]):
                predictions_df.at[row.name, timber_trait] = abs(row[timber_trait] - normalized_ground_truth_values[i])
        else:
            print(f"Warning: No ground truth found for instrument '{instrument}'")

    # Save the absolute errors to a CSV file
    predictions_df.to_csv(f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_predictions_absolute_errors.csv", index=False)

def get_MAE_per_instrument(embeddings_type, hidden_layers_suffix):
    """
    Compute Mean Absolute Error (MAE) per instrument for each timber trait.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        hidden_layers_suffix (str): Suffix for the model save path based on hidden layers.

    This function:
    1. Loads the absolute errors.
    2. For each instrument, computes the MAE for each timber trait.
    3. Adds an average column for each instrument and an average row for all instruments.
    4. Saves the MAE results to a CSV file.

    Steps:
    - Load the absolute errors.
    - For each instrument, compute the MAE for each timber trait.
    - Add an average column for each instrument and an average row for all instruments.
    - Save the MAE results to a CSV file.

    Returns:
        None: MAE results are saved to a CSV file.
    """
    print(f"Computing MAE for the models trained on {embeddings_type} with hidden layers {hidden_layers_suffix}")

    # Load the absolute errors
    absolute_errors_path = f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_predictions_absolute_errors.csv"
    absolute_errors_df = pd.read_csv(absolute_errors_path)

    # Compute MAE for each instrument and each timber trait
    mae_dict_per_instrument = {}
    for instrument in absolute_errors_df["Excluded Instrument"].unique():
        instrument_df = absolute_errors_df[absolute_errors_df["Excluded Instrument"] == instrument]
        mae_dict_per_instrument[instrument] = {}
        mae_dict_per_instrument[instrument]["Excluded Instrument"] = instrument
        for col in instrument_df.columns[2:]:  # Skip first two columns (Sample and Excluded Instrument)
            mae_dict_per_instrument[instrument][col] = instrument_df[col].mean()

    # Sort instruments alphabetically
    sorted_instruments = sorted(mae_dict_per_instrument.keys())
    mae_dict_per_instrument = {instrument: mae_dict_per_instrument[instrument] for instrument in sorted_instruments}

    # Add an "Average" column to each instrument's row
    for instrument in mae_dict_per_instrument:
        mae_dict_per_instrument[instrument]["Average"] = sum([v for k, v in mae_dict_per_instrument[instrument].items() if k != "Excluded Instrument"]) / (len(mae_dict_per_instrument[instrument]) - 1)

    # Add an "Average" row
    average_row = {"Excluded Instrument": "Average"}
    for col in mae_dict_per_instrument[list(mae_dict_per_instrument.keys())[0]].keys():
        if col != "Excluded Instrument":
            average_row[col] = sum([mae_dict_per_instrument[instrument][col] for instrument in mae_dict_per_instrument.keys()]) / len(mae_dict_per_instrument)
    mae_dict_per_instrument["Average"] = average_row

    # Convert to DataFrame and save
    mae_df = pd.DataFrame(mae_dict_per_instrument).T
    # Reorder columns to have the "Average" column first after the "Excluded Instrument" column
    cols = mae_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index("Average")))
    mae_df = mae_df[cols]

    # Save the MAE results to a CSV file
    mae_df.to_csv(f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_maes_per_instrument.csv", index=False)

def compute_correlation(embedding_types, model_hidden_layers):
    """
    Compute Pearson correlation between predicted and ground truth timber traits.

    Args:
        embedding_types (list): List of embedding types (e.g., ["clap", "vggish", "mert"]).
        model_hidden_layers (list): List of hidden layer configurations.

    This function:
    1. Loads the ground truth data.
    2. For each embedding type and hidden layer configuration, computes the correlation between predicted and ground truth values.
    3. Computes the correlation for CMTTP predictions.
    4. Saves the correlation results to a CSV file.

    Steps:
    - Load the ground truth data.
    - For each embedding type and hidden layer configuration, compute the correlation between predicted and ground truth values.
    - Compute the correlation for CMTTP predictions.
    - Save the correlation results to a CSV file.

    Returns:
        None: Correlation results are saved to a CSV file.
    """
    # Load the ground truth data
    ground_truth_path = f"data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)
    timber_traits_names = ground_truth_df.columns[2:]  # Skip first two columns (Instrument, RWC Name)

    # Create a mapping from RWC Name to ground truth values
    ground_truth_mapping = {}
    for _, row in ground_truth_df.iterrows():
        ground_truth_mapping[row["RWC Name"]] = row[timber_traits_names]  # Skip first two columns (Instrument, RWC Name)

    # Initialize a dictionary to store correlation results
    corr_dict = {}

    # Compute correlation for each embedding type and hidden layer configuration
    for embeddings_type in embedding_types:
        embeddings_type = embeddings_type + "_embeddings"
        for hidden_layers_conf in model_hidden_layers:
            match len(hidden_layers_conf):
                case 0:
                    hidden_layers_suffix = "no_hidden_layers"
                case 1:
                    hidden_layers_suffix = f"single_hidden_layer"
                case _:
                    hidden_layers_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            print(f"Computing correlation for the models trained on {embeddings_type} with hidden layers {hidden_layers_suffix}")

            # Load the predictions
            predictions_path = f"experiments/cross-validation_timbre_model/results/timbre_model_{embeddings_type}_{hidden_layers_suffix}/cross-validation_predictions.csv"
            predictions_df = pd.read_csv(predictions_path)

            # Group predictions by instrument
            instruments_predictions = predictions_df.groupby("Excluded Instrument")
            instrument_mean_preds = {}
            for instrument, instrument_df in instruments_predictions:
                instrument_mean_preds[instrument] = instrument_df[timber_traits_names].mean()  # Skip first two columns (Sample and Instrument)

            # Compute correlation for each quality column
            all_predictions = []
            all_ground_truth_values = []
            for instrument, _ in instruments_predictions:
                ground_truth_values = ground_truth_mapping[instrument]
                all_ground_truth_values.extend(ground_truth_values)
                all_predictions.extend(instrument_mean_preds[instrument])

            # Compute Pearson correlation
            corr, p_value = pearsonr(all_predictions, all_ground_truth_values)
            # Format correlation string based on p-value
            match p_value:
                case p_value if p_value < 0.01:
                    corr_str = f"{corr:.3f} **"
                case p_value if p_value < 0.05:
                    corr_str = f"{corr:.3f} *"
                case _:
                    corr_str = f"{corr:.3f}"
            corr_dict[f"{embeddings_type}_{hidden_layers_suffix}"] = corr_str

    # Compute correlation for CMTTP predictions
    cmttp_predictions_path = "models/cross-validation_timbre_model/CMTTP/CMTTP_predictions.csv"
    cmttp_predictions_df = pd.read_csv(cmttp_predictions_path)

    # Group CMTTP predictions by instrument
    cmttp_instruments_predictions = cmttp_predictions_df.groupby("Instrument")
    cmttp_instrument_mean_preds = {}
    for instrument, instrument_df in cmttp_instruments_predictions:
        cmttp_instrument_mean_preds[instrument] = instrument_df[timber_traits_names].mean()  # Skip first two columns (Sample and Instrument)

    # Compute correlation for CMTTP predictions
    cmttp_all_predictions = []
    cmttp_all_ground_truth_values = []
    for instrument, _ in cmttp_instruments_predictions:
        cmttp_ground_truth_values = ground_truth_mapping[instrument]
        cmttp_all_ground_truth_values.extend(cmttp_ground_truth_values)
        cmttp_all_predictions.extend(cmttp_instrument_mean_preds[instrument])

    # Compute Pearson correlation for CMTTP
    cmttp_corr, cmttp_p_value = pearsonr(cmttp_all_predictions, cmttp_all_ground_truth_values)
    # Format correlation string based on p-value
    match cmttp_p_value:
        case cmttp_p_value if cmttp_p_value < 0.01:
            cmttp_corr_str = f"{cmttp_corr:.3f} **"
        case cmttp_p_value if cmttp_p_value < 0.05:
            cmttp_corr_str = f"{cmttp_corr:.3f} *"
        case _:
            cmttp_corr_str = f"{cmttp_corr:.3f}"
    corr_dict["CMTTP"] = cmttp_corr_str

    # Convert correlation dictionary to DataFrame and save
    corr_df = pd.DataFrame(list(corr_dict.items()), columns=["Model", "Correlation"])
    corr_df.to_csv(f"experiments/cross-validation_timbre_model/results/cross-validation_correlations_all_models.csv", index=False)

def compute_predictions_metrics():
    """
    Compute and save all prediction metrics for timber traits.

    This function:
    1. Loads the configuration from a YAML file.
    2. For each embedding type and hidden layer configuration, computes predictions, errors, and MAE.
    3. Computes the correlation between predicted and ground truth values.

    Steps:
    - Load the configuration from a YAML file.
    - For each embedding type and hidden layer configuration, compute predictions, errors, and MAE.
    - Compute the correlation between predicted and ground truth values.

    Returns:
        None: All results are saved to CSV files.
    """
    # Load the configuration from the YAML file
    with open("experiments/cross-validation_timbre_model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    embeddings_types = config["embeddings_types"]
    model_hidden_layers = config["model_hidden_layers"]

    # Compute predictions, errors, and MAE for each embedding type and hidden layer configuration
    for embeddings_type in embeddings_types:
        embeddings_type = embeddings_type + "_embeddings"
        for hidden_layers_conf in model_hidden_layers:
            match len(hidden_layers_conf):
                case 0:
                    hidden_layers_suffix = "no_hidden_layers"
                case 1:
                    hidden_layers_suffix = f"single_hidden_layer"
                case _:
                    hidden_layers_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            # Compute predictions, errors, and MAE
            compute_predictions(embeddings_type, hidden_layers_conf, hidden_layers_suffix)
            compute_errors(embeddings_type, hidden_layers_suffix)
            get_MAE_per_instrument(embeddings_type, hidden_layers_suffix)

    # Compute correlation for all models
    compute_correlation(embeddings_types, model_hidden_layers)
