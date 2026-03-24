import pandas as pd  # Data manipulation and analysis
from tqdm import tqdm  # Progress bar for iterative tasks
import plotly.graph_objects as go  # Interactive plotting library
import numpy as np  # Numerical operations
import os  # Operating system interfaces for directory and file operations
import yaml  # YAML file handling

def plot_radar_chart(embedding_type: str, hidden_layer_suffix: str, save_folder: str, verbose: bool = False):
    """
    Plot and save a radar chart comparing ground truth and predicted timbre trait values for each instrument.

    Args:
        embedding_type (str): Type of embeddings used for training the model (e.g., "clap", "vggish").
        hidden_layer_suffix (str): Suffix indicating the hidden layer configuration (e.g., "no_hidden_layers", "single_hidden_layer").
        save_folder (str): Folder path to save the radar chart images.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    This function:
    1. Loads ground truth timbre trait values and predicted values.
    2. Computes 95% confidence intervals for both ground truth and predicted values.
    3. Creates a radar chart comparing ground truth and predicted values with confidence intervals.
    4. Saves the radar chart as a PNG file.

    Steps:
    - Load ground truth and predicted values.
    - Compute 95% confidence intervals for ground truth and predicted values.
    - Create a radar chart with ground truth, predicted values, and their confidence intervals.
    - Save the radar chart to the specified folder.

    Returns:
        None: Radar charts are saved as PNG files.
    """
    # Load ground truth data from CSV
    ground_truth_csv_path = f"data/Reymore/timbre_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_csv_path)

    # Get instrument names
    instruments_names = ground_truth_df['RWC Name'].tolist()

    # Timbre traits names in the order defined by Lindsey Reymore
    timbre_traits_names = [
        "sparkling-brilliant", "focused-compact", "pure-clear", "open",
        "ringing-long_decay", "resonant-vibrant", "sustained-even", "soft-singing",
        "watery-fluid", "muted-veiled", "hollow", "woody",
        "airy-breathy", "nasal-reedy", "raspy-grainy", "rumbling-low",
        "direct-loud", "percussive", "shrill-noisy", "brassy-metallic", "sparkling-brilliant"  # Close the polygons
    ]

    # Load predicted values
    predicted_values_path = f"experiments/cross-validation_timbre_model/results/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions.csv"
    predicted_values_df = pd.read_csv(predicted_values_path)

    # Load human ratings to compute 95% confidence intervals
    ratings_path = "data/Reymore/timbre_traits_human_ratings.csv"
    ratings_df = pd.read_csv(ratings_path)

    # Select relevant columns: Instrument and timbre traits
    ratings_df = ratings_df[['Instrument'] + timbre_traits_names]
    ground_truth_values_confidence_intervals = {}

    for instrument in tqdm(instruments_names, desc=f"Computing radar charts for model trained on {embedding_type} with {hidden_layer_suffix}"):
        # Get the Reymore name for the instrument
        reymore_name = ground_truth_df[ground_truth_df['RWC Name'] == instrument]['Instrument'].iloc[0]
        instrument_ratings = ratings_df[ratings_df['Instrument'] == reymore_name]

        # Compute 95% confidence intervals of human ratings for each timbre trait
        ground_truth_values_confidence_intervals[instrument] = []
        for timbre_trait in timbre_traits_names:
            values = instrument_ratings[timbre_trait].values
            std = np.std(values, ddof=1)
            predicted_values_confidence_interval = 1.96 * std / np.sqrt(len(values))
            ground_truth_values_confidence_intervals[instrument].append(predicted_values_confidence_interval)

        # Get ground truth values
        ground_truth_row = ground_truth_df[ground_truth_df['RWC Name'] == instrument]
        ground_truth_values = ground_truth_row[timbre_traits_names].values[0].tolist()  # Skip the "Instrument" column

        # Extract predicted values for the selected instrument
        predicted_values_row = predicted_values_df[predicted_values_df['Excluded Instrument'] == instrument]
        predicted_values = predicted_values_row[timbre_traits_names].values  # Skip the "Instrument" column
        predicted_values = predicted_values * 6 + 1  # Denormalize predicted values

        # Compute 95% confidence interval over the predicted values
        std = np.std(predicted_values, axis=0)
        predicted_values_confidence_interval = 1.96 * std / np.sqrt(predicted_values.shape[0])  # 95% confidence interval

        # Mean predicted values for the selected instrument
        predicted_values = np.mean(predicted_values, axis=0)
        predicted_values = np.append(predicted_values, predicted_values[0])  # Close the polygon
        predicted_values_confidence_interval = np.append(predicted_values_confidence_interval, predicted_values_confidence_interval[0])  # Close the polygon

        # Create radar chart
        fig = go.Figure()

        # Add ground truth trace (green)
        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values,
            theta=timbre_traits_names,
            name='Ground Truth',
            line=dict(color='rgba(44, 160, 44, 1)'),
        ))

        # Add ground truth + confidence interval trace (invisible line)
        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values + np.array(ground_truth_values_confidence_intervals[instrument]),
            theta=timbre_traits_names,
            name='Ground Truth + Confidence Interval',
            line=dict(color='rgba(44, 160, 44, 0)'),
            showlegend=False,
        ))

        # Add ground truth - confidence interval trace (filled area)
        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values - np.array(ground_truth_values_confidence_intervals[instrument]),
            theta=timbre_traits_names,
            name='Ground Truth - Confidence Interval',
            line=dict(color='rgba(44, 160, 44, 0)'),
            showlegend=False,
            fill='tonext',
            fillcolor='rgba(44, 160, 44, 0.4)',
        ))

        # Add predicted values trace (blue)
        fig.add_trace(go.Scatterpolar(
            r=predicted_values,
            theta=timbre_traits_names,
            name='Predicted',
            line=dict(color='rgba(31, 119, 180, 1)'),
        ))

        # Add predicted values + confidence interval trace (invisible line)
        fig.add_trace(go.Scatterpolar(
            r=predicted_values + predicted_values_confidence_interval,
            theta=timbre_traits_names,
            name='Predicted + Confidence Interval',
            line=dict(color='rgba(31, 119, 180, 0)'),
            showlegend=False,
        ))

        # Add predicted values - confidence interval trace (filled area)
        fig.add_trace(go.Scatterpolar(
            r=predicted_values - predicted_values_confidence_interval,
            theta=timbre_traits_names,
            name='Predicted - Confidence Interval',
            line=dict(color='rgba(31, 119, 180, 0)'),
            showlegend=False,
            fill='tonext',
            fillcolor='rgba(31, 119, 180, 0.4)',
        ))

        # Update layout for better visualization
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, max(max(ground_truth_values), max(predicted_values)) * 1.1]
                )),
            showlegend=True,
            title={
                "text": f"Predicted Timbre Traits Profile of {instrument} by the model trained without {instrument} samples",
                'x': 0.5,  # Centers the title
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                yanchor="middle",  # Anchor the legend vertically in the middle
                y=0.5,  # Position the legend at the vertical middle
                xanchor="left",  # Anchor the legend to the left of its position
                x=1.05,  # Position the legend just outside the right edge
            ),
            autosize=False,  # Disable autosize to use the specified dimensions
            width=900,
            height=600
        )

        # Save the plot
        save_path = os.path.join(save_folder, f"radar_chart_excluded_instrument_{instrument.replace(' ', '_')}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        if verbose:
            print(f"Radar chart saved to {save_path}")

def plot_all_instruments_radar_charts():
    """
    Plot radar charts for all instruments and all model configurations.

    This function:
    1. Loads the configuration from a YAML file.
    2. For each embedding type and hidden layer configuration, plots radar charts for all instruments.

    Steps:
    - Load the configuration from a YAML file.
    - For each embedding type and hidden layer configuration, call the `plot_radar_chart` function.

    Returns:
        None: Radar charts are saved as PNG files.
    """
    print("Plotting radar charts for all instruments...")

    # Load configuration from YAML file
    with open("experiments/cross-validation_timbre_model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters
    embeddings_types = config["embeddings_types"]
    model_hidden_layers = config["model_hidden_layers"]

    # Plot radar charts for each embedding type and hidden layer configuration
    for embedding_type in embeddings_types:
        embedding_type = embedding_type + "_embeddings"
        for hidden_layers_conf in model_hidden_layers:
            match len(hidden_layers_conf):
                case 0:
                    hidden_layer_suffix = "no_hidden_layers"
                case 1:
                    hidden_layer_suffix = f"single_hidden_layer"
                case _:
                    hidden_layer_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            plot_radar_chart(embedding_type, hidden_layer_suffix, save_folder=f"experiments/cross-validation_timbre_model/results/timbre_model_{embedding_type}_{hidden_layer_suffix}/")
