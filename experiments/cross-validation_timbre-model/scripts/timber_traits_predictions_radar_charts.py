import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import yaml

def plot_radar_chart(embedding_type: str, hidden_layer_suffix: str, save_folder: str):
    """
    Plot and save a radar chart comparing ground truth and predicted values for a given instrument.

    Args:
        instrument (str): Name of the instrument to plot.
        data_dict (dict): Dictionary of shape {Instrument: list_of_values}.
        csv_path (str): Path to the ground_truth.csv file.
        save_path (str): Path to save the radar chart PNG file.
    """
    # Load ground truth data from CSV
    ground_truth_csv_path = f"data/Reymore/timber_traits_ground_truth.csv"
    ground_truth_df = pd.read_csv(ground_truth_csv_path)

    # Get instruments names
    instruments_names = ground_truth_df['RWC Name'].tolist()

    # Extract ground truth values for the selected instrument
    timber_traits_names = ground_truth_df.columns[2:].tolist() # Get the names of the timber_traits (excluding the first column which is 'Instrument')

    # Load predicted values
    predicted_values_path = f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/cross-validation_predictions.csv"
    predicted_values_df = pd.read_csv(predicted_values_path)

    # Load human ratings to compute 95% confidence intervals
    ratings_path = "data/Reymore/timber_traits_human_ratings.csv"

    # Read the CSV files
    ratings_df = pd.read_csv(ratings_path)

    # Select relevant columns: participant, Instrument, and timber_traits (6th column onwards)
    ratings_df = ratings_df[['Instrument'] + timber_traits_names]
    ground_truth_values_confidence_intervals = {}

    for instrument in instruments_names:
        reymore_name = ground_truth_df[ground_truth_df['RWC Name'] == instrument]['Instrument'].iloc[0]
        instrument_ratings = ratings_df[ratings_df['Instrument'] == reymore_name]

        # Compute of the 95% confidence intervals of human ratings for each timber trait
        ground_truth_values_confidence_intervals[instrument] = []
        for timber_trait in timber_traits_names:
            values = instrument_ratings[timber_trait].values
            std = np.std(values, ddof=1)
            predicted_values_confidence_interval = 1.96 * std / np.sqrt(len(values))
            ground_truth_values_confidence_intervals[instrument].append(predicted_values_confidence_interval)
        
        # Get ground truth values
        ground_truth_values = ground_truth_df[ground_truth_df['Instrument'] == instrument].iloc[0][2:].values.tolist() # Skip the "Instrument" column

        # Extract predicted values for the selected instrument
        predicted_values = predicted_values_df[predicted_values_df['Instrument'] == instrument].iloc[0][2:].values.tolist() # Skip the "Instrument" column
        predicted_values = np.mean(predicted_values)

        # Compute 95% confidence interval over the predicted_values
        std = np.std(predicted_values, axis=0)
        predicted_values_confidence_interval = 1.96 * std / np.sqrt(predicted_values.shape[0])  # 95% confidence interval

        # Mean predicted values for the selected instrument
        predicted_values = np.mean(predicted_values, axis=0)

        # Create radar chart
        fig = go.Figure()

        # tab10 green: 44, 160, 44
        # Add ground truth trace
        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values,
            theta=timber_traits_names,
            name='Ground Truth',
            line=dict(color='rgba(44, 160, 44, 1)'),
        ))

        # Add 95% confidence interval predicted values trace
        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values + np.array(ground_truth_values_confidence_intervals[instrument]),
            theta=timber_traits_names,
            name='Ground Truth + Confidence Interval',
            line=dict(color='rgba(44, 160, 44, 0)'),
            showlegend=False,
        ))

        fig.add_trace(go.Scatterpolar(
            r=ground_truth_values - np.array(ground_truth_values_confidence_intervals[instrument]),
            theta=timber_traits_names,
            name='Ground Truth - Confidence Interval',
            line=dict(color='rgba(44, 160, 44, 0)'),
            showlegend=False,
            fill = 'tonext',
            fillcolor='rgba(44, 160, 44, 0.4)',
        ))

        # tab10 blue: 31, 119, 180
        # Add predicted values trace
        fig.add_trace(go.Scatterpolar(
            r=predicted_values,
            theta=timber_traits_names,
            name='Predicted',
            line=dict(color='rgba(31, 119, 180, 1)'),
        ))

        # Add 95% confidence interval predicted values trace
        fig.add_trace(go.Scatterpolar(
            r=predicted_values + predicted_values_confidence_interval,
            theta=timber_traits_names,
            name='Predicted + Confidence Interval',
            line=dict(color='rgba(31, 119, 180, 0)'),
            showlegend=False,
        ))

        fig.add_trace(go.Scatterpolar(
            r=predicted_values - predicted_values_confidence_interval,
            theta=timber_traits_names,
            name='Predicted - Confidence Interval',
            line=dict(color='rgba(31, 119, 180, 0)'),
            showlegend=False,
            fill = 'tonext',
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
                "text": f"Predicted Timber Traits Profile of {instrument} by the model trained without {instrument} samples",
                'x': 0.5,  # Centers the title
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                    yanchor="middle",  # Anchor the legend vertically in the middle
                    y=0.5,            # Position the legend at the vertical middle
                    xanchor="left",   # Anchor the legend to the left of its position
                    x=1.05,           # Position the legend just outside the right edge
                ),
            autosize=True # Disable autosize to use the specified dimensions
        )

        # Save the plot
        save_path = os.path.join(save_folder, f"radar_chart_excluded_instrument_{instrument.replace(' ', '_')}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        print(f"Radar chart saved to {save_path}")

def plot_all_instruments_radar_charts():

    # Load config.yaml
    with open("experiments/cross-validation_timbre-model/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embeddings_types = config["embeddings_types"]
    model_hidden_layers = config["model_hidden_layers"]

    for embedding_type in embeddings_types:
        for hidden_layers_conf in model_hidden_layers:
            
            match len(hidden_layers_conf):
                case 0:
                    hidden_layer_suffix = "no_hidden_layers"
                case 1:
                    hidden_layer_suffix = f"single_hidden_layer"
                case _:
                    hidden_layer_suffix = f"{len(hidden_layers_conf)}_hidden_layers"

            plot_radar_chart(embedding_type, hidden_layer_suffix, save_path=f"experiments/cross-validation_timbre-model/timbre_model_{embedding_type}_{hidden_layer_suffix}/")