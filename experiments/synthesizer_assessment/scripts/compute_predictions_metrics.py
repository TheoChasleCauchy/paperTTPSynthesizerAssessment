import pandas as pd  # Data manipulation and analysis
import os  # Operating system interfaces for directory and file operations
from timbre_mlp import TimbreMLP  # Custom MLP model for timbre prediction
from samples_dataset import SamplesDataset  # Custom dataset class for handling samples
from tqdm import tqdm  # Progress bar for iterative tasks
import torch  # PyTorch for tensor operations and device management
from scipy.linalg import sqrtm  # Square root of a matrix for FAD computation
import numpy as np  # Numerical operations

def compute_predictions_on_TokenSynth(embeddings_type, model_hidden_layers, hidden_layers_suffix):
    """
    Compute predictions for synthesized audio samples using a pre-trained TimbreMLP model.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        model_hidden_layers (list): List of hidden layer sizes for the TimbreMLP model.
        hidden_layers_suffix (str): Suffix for the model save path based on hidden layers.

    This function:
    1. Determines the input size based on the embedding type.
    2. For each condition type (text, audio, text_audio), loads the synthesized samples metadata.
    3. For each sample, loads the pre-trained model and computes predictions.
    4. Saves the predictions to a CSV file for each condition type.

    Steps:
    - Set the input size based on the embedding type.
    - For each condition type, load the metadata and initialize the model.
    - For each sample, compute predictions and save them to a DataFrame.
    - Save the predictions to a CSV file.

    Returns:
        None: Predictions are saved to CSV files.
    """
    embeddings_type = embeddings_type + "_embeddings"

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

    output_size = 20  # 20 timbre traits
    model_save_path = f"models/synthesizer_assessment/timbre_model_{embeddings_type}_{hidden_layers_suffix}"

    # Iterate over each condition type
    for condition_type in ["text", "audio", "text_audio"]:
        dataset_path = f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}/embeddings_labels.csv"
        dataset = pd.read_csv(dataset_path)
        timbre_traits_names = dataset.columns[2:].tolist()
        df = []

        # Compute predictions for each sample
        for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc=f"Computing predictions for all samples for {condition_type}"):
            instrument = row.Instrument

            # Load the pre-trained model
            model = TimbreMLP.load_model(
                f"{model_save_path}/timbre_mlp.pth",
                input_size=input_size,
                hidden_sizes=model_hidden_layers,
                output_size=output_size,
            )

            # Create a DataLoader for the current sample
            _, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

            # Evaluate the model and get predictions
            _, predicted_values, _ = model.evaluate_model(evalDataloader)

            # Append predictions to the DataFrame
            df.append({
                "Sample": row.Path,
                "Instrument": instrument,
                **{timbre_trait: predicted_values[:, timbre_trait_id].item() for timbre_trait_id, timbre_trait in enumerate(timbre_traits_names)}
            })

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(df)

        # Save the predictions to a CSV file
        os.makedirs(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/", exist_ok=True)
        df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv", index=False)

def compute_predictions_on_RWC(embeddings_type, model_hidden_layers, hidden_layers_suffix):
    """
    Compute predictions for RWC audio samples using a pre-trained TimbreMLP model.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        model_hidden_layers (list): List of hidden layer sizes for the TimbreMLP model.
        hidden_layers_suffix (str): Suffix for the model save path based on hidden layers.

    This function:
    1. Determines the input size based on the embedding type.
    2. Loads the RWC samples metadata.
    3. For each sample, loads the pre-trained model and computes predictions.
    4. Saves the predictions to a CSV file.

    Steps:
    - Set the input size based on the embedding type.
    - Load the metadata and initialize the model.
    - For each sample, compute predictions and save them to a DataFrame.
    - Save the predictions to a CSV file.

    Returns:
        None: Predictions are saved to a CSV file.
    """
    embeddings_type = embeddings_type + "_embeddings"

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

    output_size = 20  # 20 timbre traits
    model_save_path = f"models/synthesizer_assessment/timbre_model_{embeddings_type}_{hidden_layers_suffix}"

    # Load the RWC samples metadata
    dataset_path = f"data/metadata/RWC/clap_embeddings/{embeddings_type}_labels.csv"
    dataset = pd.read_csv(dataset_path)
    timbre_traits_names = dataset.columns[2:].tolist()
    df = []

    # Compute predictions for each sample
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc=f"Computing predictions for all samples of RWC"):
        instrument = row.Instrument

        # Load the pre-trained model
        model = TimbreMLP.load_model(
            f"{model_save_path}/timbre_mlp.pth",
            input_size=input_size,
            hidden_sizes=model_hidden_layers,
            output_size=output_size,
        )

        # Create a DataLoader for the current sample
        _, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

        # Evaluate the model and get predictions
        _, predicted_values, _ = model.evaluate_model(evalDataloader)

        # Append predictions to the DataFrame
        df.append({
            "Sample": row.Path,
            "Instrument": instrument,
            **{timbre_trait: predicted_values[:, timbre_trait_id].item() for timbre_trait_id, timbre_trait in enumerate(timbre_traits_names)}
        })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(df)

    # Save the predictions to a CSV file
    os.makedirs(f"experiments/synthesizer_assessment/results/RWC/", exist_ok=True)
    df.to_csv(f"experiments/synthesizer_assessment/results/RWC/RWC_predictions.csv", index=False)

def compute_errors():
    """
    Compute absolute errors between predicted and ground truth timbre traits.

    This function:
    1. For each condition type, loads the predictions and ground truth data.
    2. For each sample, computes the absolute difference between predicted and ground truth values.
    3. Saves the absolute errors to a CSV file for each condition type.

    Steps:
    - For each condition type, load the predictions and ground truth data.
    - For each sample, compute the absolute difference between predicted and ground truth values.
    - Save the absolute errors to a CSV file.

    Returns:
        None: Absolute errors are saved to CSV files.
    """
    for condition_type in ["text", "audio", "text_audio"]:
        predictions_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv"
        predictions_df = pd.read_csv(predictions_path)

        ground_truth_path = f"data/Reymore/timbre_traits_ground_truth.csv"
        ground_truth_df = pd.read_csv(ground_truth_path)

        # Create a mapping from RWC Name to ground truth values
        ground_truth_mapping = {}
        for _, row in ground_truth_df.iterrows():
            ground_truth_mapping[row["RWC Name"]] = row[2:]

        # Compute absolute errors for each sample
        for _, row in predictions_df.iterrows():
            instrument = row["Instrument"]
            if instrument in ground_truth_mapping:
                ground_truth_values = ground_truth_mapping[instrument]
                normalized_ground_truth_values = (ground_truth_values - 1) / 6.0
                for i, timbre_trait in enumerate(ground_truth_df.columns[2:]):
                    predictions_df.at[row.name, timbre_trait] = abs(row[timbre_trait] - normalized_ground_truth_values[i])
            else:
                print(f"Warning: No ground truth found for instrument '{instrument}'")

        # Save the absolute errors to a CSV file
        predictions_df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_absolute_errors.csv", index=False)

def get_MAE_per_instrument():
    """
    Compute Mean Absolute Error (MAE) per instrument for each timbre trait.

    This function:
    1. For each condition type, loads the absolute errors.
    2. For each instrument, computes the MAE for each timbre trait.
    3. Adds an average column for each instrument and an average row for all instruments.
    4. Saves the MAE results to a CSV file for each condition type.

    Steps:
    - For each condition type, load the absolute errors.
    - For each instrument, compute the MAE for each timbre trait.
    - Add an average column for each instrument and an average row for all instruments.
    - Save the MAE results to a CSV file.

    Returns:
        None: MAE results are saved to CSV files.
    """
    for condition_type in ["text", "audio", "text_audio"]:
        absolute_errors_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_absolute_errors.csv"
        absolute_errors_df = pd.read_csv(absolute_errors_path)
        instrument_names = absolute_errors_df["Instrument"].unique()

        # Compute MAE for each instrument and each timbre trait
        mae_dict_per_instrument = {}
        for instrument in instrument_names:
            instrument_df = absolute_errors_df[absolute_errors_df["Instrument"] == instrument]
            mae_dict_per_instrument[instrument] = {}
            mae_dict_per_instrument[instrument]["Instrument"] = instrument
            for col in instrument_df.columns[2:]:
                mae_dict_per_instrument[instrument][col] = instrument_df[col].mean()

        # Sort instruments alphabetically
        sorted_instruments = sorted(mae_dict_per_instrument.keys())
        mae_dict_per_instrument = {instrument: mae_dict_per_instrument[instrument] for instrument in sorted_instruments}

        # Add an "Average" column to each instrument's row
        for instrument in mae_dict_per_instrument:
            average = sum([v for k, v in mae_dict_per_instrument[instrument].items() if k != "Instrument"]) / (len(mae_dict_per_instrument[instrument]) - 1)
            mae_dict_per_instrument[instrument]["Average"] = average

        # Add an "Average" row
        average_row = {"Instrument": "Average"}
        for col in mae_dict_per_instrument[list(mae_dict_per_instrument.keys())[0]].keys():
            if col != "Instrument" and col != "Average":
                average_mae = sum([mae_dict_per_instrument[instrument][col] for instrument in mae_dict_per_instrument.keys()]) / len(mae_dict_per_instrument)
                average_std = np.mean(absolute_errors_df[absolute_errors_df["Instrument"].isin(instrument_names)][col].std())
                average_row[col] = f"{average_mae:.3f} ± {average_std:.3f}"

        average_mae = np.mean(absolute_errors_df[absolute_errors_df.columns[2:]].mean())
        average_std = np.mean(absolute_errors_df[absolute_errors_df.columns[2:]].std())
        average_row["Average"] = f"{average_mae:.3f} ± {average_std:.3f}"

        mae_dict_per_instrument["Average"] = average_row

        # Convert to DataFrame and reorder columns
        mae_df = pd.DataFrame(mae_dict_per_instrument).T
        cols = mae_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Average")))
        mae_df = mae_df[cols]

        # Save the MAE results to a CSV file
        mae_df.to_csv(f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_maes_per_instrument.csv", index=False)

def compute_fad_RWC_Synth(embeddings_type: str):
    """
    Compute Fréchet Audio Distance (FAD) between RWC and synthesized audio embeddings.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").

    This function:
    1. For each condition type, loads the embeddings for RWC and synthesized audio.
    2. Computes the FAD score for each instrument.
    3. Saves the FAD scores to a CSV file.

    Steps:
    - For each condition type, load the embeddings for RWC and synthesized audio.
    - Compute the FAD score for each instrument.
    - Save the FAD scores to a CSV file.

    Returns:
        None: FAD scores are saved to a CSV file.
    """
    embeddings_type = embeddings_type + "_embeddings"

    def load_embeddings(folder_path, instrument_name):
        """
        Load embeddings for a specific instrument from a folder.

        Args:
            folder_path (str): Path to the folder containing embeddings.
            instrument_name (str): Name of the instrument.

        Returns:
            torch.Tensor: Stacked embeddings for the instrument.
        """
        embeddings = []
        for file in os.listdir(folder_path):
            if instrument_name in file and file.endswith('.pt'):
                emb = torch.load(os.path.join(folder_path, file))
                embeddings.append(emb)
        return torch.stack(embeddings)

    def compute_fad(mu1, sigma1, mu2, sigma2):
        """
        Compute the Fréchet Audio Distance (FAD) between two sets of embeddings.

        Args:
            mu1 (torch.Tensor): Mean of the first set of embeddings.
            sigma1 (torch.Tensor): Covariance matrix of the first set of embeddings.
            mu2 (torch.Tensor): Mean of the second set of embeddings.
            sigma2 (torch.Tensor): Covariance matrix of the second set of embeddings.

        Returns:
            torch.Tensor: FAD score.
        """
        diff = mu1 - mu2
        covmean = torch.from_numpy(sqrtm(sigma1 @ sigma2))
        if torch.is_complex(covmean):
            covmean = covmean.real
        fad = torch.sum(diff**2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fad

    condition_types = ["text", "audio", "text_audio"]
    fads = {cond: {} for cond in condition_types}

    for condition_type in condition_types:
        metadata_path = f"experiments/synthesizer_assessment/results/{condition_type}_conditioned_synthesis/{condition_type}_predictions.csv"
        metadata_df = pd.read_csv(metadata_path)
        instrument_names = metadata_df["Instrument"].unique()

        for instrument in tqdm(instrument_names, total=len(instrument_names), desc=f"Processing {condition_type}"):
            emb1 = load_embeddings(f"data/RWC/embeddings/{embeddings_type}/", instrument_name=instrument)
            emb2 = load_embeddings(f"data/TokenSynth/Embeddings/{condition_type}_conditioned_synthesis/{embeddings_type}/", instrument_name=instrument)

            mu1, sigma1 = torch.mean(emb1, axis=0), torch.cov(emb1.T)
            mu2, sigma2 = torch.mean(emb2, axis=0), torch.cov(emb2.T)

            fad_score = compute_fad(mu1, sigma1, mu2, sigma2)
            fads[condition_type][instrument] = fad_score.item()

        fads[condition_type]["Mean"] = sum(fads[condition_type].values()) / len(fads[condition_type]) if fads[condition_type] else 0

    # Save FAD scores to a CSV file
    fads_df = pd.DataFrame.from_dict(fads, orient='index')
    mean_col = fads_df.pop("Mean")
    fads_df.insert(0, "Mean", mean_col)
    fads_df.to_csv("experiments/synthesizer_assessment/results/fad_results.csv", index_label="Condition Type")

def fad_mae_table():
    """
    Combine FAD and MAE results into a single table.

    This function:
    1. Loads the FAD results and MAE results for each condition type.
    2. Combines the FAD and MAE results into a single table.
    3. Saves the combined table to a CSV file.

    Steps:
    - Load the FAD results and MAE results for each condition type.
    - Combine the FAD and MAE results into a single table.
    - Save the combined table to a CSV file.

    Returns:
        None: Combined table is saved to a CSV file.
    """
    fad_results = pd.read_csv("experiments/synthesizer_assessment/results/fad_results.csv")
    text_mae = pd.read_csv("experiments/synthesizer_assessment/results/text_conditioned_synthesis/text_maes_per_instrument.csv")
    audio_mae = pd.read_csv("experiments/synthesizer_assessment/results/audio_conditioned_synthesis/audio_maes_per_instrument.csv")
    text_audio_mae = pd.read_csv("experiments/synthesizer_assessment/results/text_audio_conditioned_synthesis/text_audio_maes_per_instrument.csv")

    # Get the first two columns of the FAD table
    fad_mae_table = fad_results[fad_results.columns[:2]]

    # Add a "MAE" column for each condition type
    mae_column = []
    for condition_type in ["text", "audio", "text_audio"]:
        if condition_type == "text":
            mae_column.append(text_mae["Average"].iloc[-1])
        elif condition_type == "audio":
            mae_column.append(audio_mae["Average"].iloc[-1])
        elif condition_type == "text_audio":
            mae_column.append(text_audio_mae["Average"].iloc[-1])

    fad_mae_table["MAE"] = pd.DataFrame(mae_column)
    fad_mae_table.to_csv("experiments/synthesizer_assessment/results/fad_mae_results.csv", index=False)

def compute_predictions_metrics(embeddings_type: str, model_hidden_layers: list):
    """
    Compute and save all prediction metrics for synthesized and RWC audio samples.

    Args:
        embeddings_type (str): Type of embeddings (e.g., "clap", "vggish", "mert").
        model_hidden_layers (list): List of hidden layer sizes for the TimbreMLP model.

    This function:
    1. Determines the hidden layers suffix based on the number of hidden layers.
    2. Computes predictions for synthesized and RWC audio samples.
    3. Computes absolute errors between predictions and ground truth.
    4. Computes MAE per instrument.
    5. Computes FAD between RWC and synthesized audio embeddings.
    6. Combines FAD and MAE results into a single table.

    Steps:
    - Determine the hidden layers suffix.
    - Compute predictions for synthesized and RWC audio samples.
    - Compute absolute errors between predictions and ground truth.
    - Compute MAE per instrument.
    - Compute FAD between RWC and synthesized audio embeddings.
    - Combine FAD and MAE results into a single table.

    Returns:
        None: All results are saved to CSV files.
    """
    # Determine the hidden layers suffix
    match len(model_hidden_layers):
        case 0:
            hidden_layers_suffix = "no_hidden_layers"
        case 1:
            hidden_layers_suffix = "single_hidden_layer"
        case _:
            hidden_layers_suffix = f"{len(model_hidden_layers)}_hidden_layers"

    # Compute predictions and metrics
    compute_predictions_on_TokenSynth(embeddings_type, model_hidden_layers, hidden_layers_suffix)
    compute_predictions_on_RWC(embeddings_type, model_hidden_layers, hidden_layers_suffix)
    compute_errors()
    get_MAE_per_instrument()
    compute_fad_RWC_Synth(embeddings_type)
    fad_mae_table()
