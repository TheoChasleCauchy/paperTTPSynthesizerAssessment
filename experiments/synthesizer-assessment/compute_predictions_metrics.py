import pandas as pd
import os
from timbre_mlp import TimbreMLP
from samples_dataset import SamplesDataset
from tqdm import tqdm
import yaml

def compute_predictions(embeddings_type, model_hidden_layers, hidden_layers_suffix):

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
        case _:
            raise ValueError(f"Unsupported embedding type: {embeddings_type}")
    output_size = 20 # 20 timber traits
    
    model_save_path = f"models/synthesizer-assessment/timbre_model_{embeddings_type}_{hidden_layers_suffix}/"
    
    for condition_type in ["text_conditioned", "audio_conditioned", "text_audio_conditioned"]:
        dataset_path = f"resources/metadata/Synth/{condition_type}_embeddings_labels.csv"
        dataset = pd.read_csv(dataset_path)
        timbre_traits_names = dataset.columns[2:].tolist() 
        df = []

        # Compute the predicted values for each sample (row) of the dataset and add it to the dataFrame
        for row in tqdm(dataset.itertuples(index=False), total=len(dataset), desc=f"Computing predictions for all samples for {condition_type}"):
            instrument = row.Instrument

            model = TimbreMLP.load_model(
                f"{model_save_path}/timbre_mlp.pth",
                input_size=input_size,
                hidden_sizes=model_hidden_layers,
                output_size=output_size,
            )
            
            evalDataset, evalDataloader = SamplesDataset.create_dataloader(df=pd.DataFrame([row]), batch_size=1)

            eval_loss, predicted_values, _ = model.evaluate_model(evalDataloader)

            df.append({
                "Sample": row.Path,
                "Instrument": instrument,
                **{timber_trait: predicted_values[:,timber_trait_id].item() for timber_trait_id, timber_trait in enumerate(timbre_traits_names)}  # Skip first two columns (Sample and Excluded Instrument)
            })
        
        df = pd.DataFrame(df)
        df.to_csv(f"metrics/Synth/{condition_type}_predictions.csv", index=False)

def compute_errors():
    for condition_type in ["text_conditioned", "audio_conditioned", "text_audio_conditioned"]:
        predictions_path = f"metrics/Synth/{condition_type}_predictions.csv"

        # Read the predictions CSV file
        predictions_df = pd.read_csv(predictions_path)

        # Get the ground truth CSV
        ground_truth_path = f"resources/metadata/qualities_ground_truth.csv"
        ground_truth_df = pd.read_csv(ground_truth_path)

        # for each row for each timber_trait column, replace the value by the absolute difference between this value and the ground_truth value
        ground_truth_mapping = {}
        for _, row in ground_truth_df.iterrows():
            ground_truth_mapping[row["RWC Name"]] = row[2:]  # Skip first two columns (Excluded Instrument, RWC Name)

        for _, row in predictions_df.iterrows():
            instrument = row["Excluded Instrument"]
            if instrument in ground_truth_mapping:
                ground_truth_values = ground_truth_mapping[instrument]
                normalized_ground_truth_values = (ground_truth_values - 1) / 6.0  # Normalize to [0,1]
                for i, timber_trait in enumerate(ground_truth_df.columns[2:]):
                    predictions_df.at[row.name, timber_trait] = abs(row[timber_trait] - normalized_ground_truth_values[i])
            else:
                print(f"Warning: No ground truth found for instrument '{instrument}'")

        # Save the updated predictions with absolute differences
        predictions_df.to_csv(f"metrics/Synth/{condition_type}_absolute_errors.csv", index=False)

def get_MAE_per_instrument():
    for condition_type in ["text_conditioned", "audio_conditioned", "text_audio_conditioned"]:
        absolute_errors_path = f"metrics/Synth/{condition_type}_absolute_errors.csv"

        # Read the predictions CSV file
        absolute_errors_df = pd.read_csv(absolute_errors_path)

        # Compute MAE for each instrument and each timber_trait column
        mae_dict_per_instrument = {}
        for instrument in absolute_errors_df["Excluded Instrument"].unique():
            instrument_df = absolute_errors_df[absolute_errors_df["Excluded Instrument"] == instrument]
            mae_dict_per_instrument[instrument] = {}
            mae_dict_per_instrument[instrument]["Excluded Instrument"] = instrument
            for col in instrument_df.columns[2:]:  # Skip first two columns (Sample and Excluded Instrument)
                mae_dict_per_instrument[instrument][col] = instrument_df[col].mean()

        # Reorder rows by the alphabetical order on the "Excluded Instrument" column
        sorted_instruments = sorted(mae_dict_per_instrument.keys())
        mae_dict_per_instrument = {instrument: mae_dict_per_instrument[instrument] for instrument in sorted_instruments}

        # Add an "Average" column to each instrument's row as the first column
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
        # Reorder columns to have Average column first after the Excluded Instrument column
        cols = mae_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Average")))
        mae_df = mae_df[cols]

        mae_df.to_csv(f"metrics/Synth/{condition_type}_maes_per_instrument.csv", index=False)

def compute_predictions_metrics():

    # Load config.yaml
    with open("experiments/cross-validation_timbre-model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_type = config["embeddings_type"]
    model_hidden_layers = config["model_hidden_layers"]

    match len(model_hidden_layers):
        case 0:
            hidden_layers_suffix = "no_hidden_layers"
        case 1:
            hidden_layers_suffix = f"single_hidden_layer"
        case _:
            hidden_layers_suffix = f"{len(model_hidden_layers)}_hidden_layers"

    compute_predictions(embeddings_type, model_hidden_layers, hidden_layers_suffix)
    compute_errors(embeddings_type, hidden_layers_suffix)
    get_MAE_per_instrument(embeddings_type, hidden_layers_suffix)