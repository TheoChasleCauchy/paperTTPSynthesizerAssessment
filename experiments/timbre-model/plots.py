import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np

def write_latex_radar_chart(instrument: str, data_dict: dict, save_path: str = "radar_chart.tex"):
    """
    Generate LaTeX code for a radar chart and save it to a .tex file.

    Args:
        instrument (str): Name of the instrument to plot.
        data_dict (dict): Dictionary of shape {Instrument: list_of_values}.
        save_path (str): Path to save the LaTeX file.
    """
    # Load ground truth data from CSV
    csv_path = "./resources/metadata/qualities_ground_truth.csv"
    ground_truth_df = pd.read_csv(csv_path)

    # Extract ground truth values for the selected instrument
    ground_truth_row = ground_truth_df[ground_truth_df["RWC Name"] == instrument].iloc[0]
    reymore_name = ground_truth_row["Instrument"]
    ground_truth_values = ground_truth_row[2:].values.tolist()
    qualities_names = ground_truth_df.columns[2:].tolist()

    # Get 95% confidence intervals of the ground truth values
    ratings_path = "./resources/metadata/qualities_ratings.csv"
    ratings_df = pd.read_csv(ratings_path)
    ratings_df = ratings_df[['Instrument'] + qualities_names]
    instrument_ratings = ratings_df[ratings_df['Instrument'] == reymore_name]
    ground_truth_values_intervals = []
    for quality in qualities_names:
        values = instrument_ratings[quality].values
        std = np.std(values, ddof=1)
        confidence_interval = 1.96 * std / np.sqrt(len(values))
        ground_truth_values_intervals.append(confidence_interval)

    # Extract predicted values for the selected instrument
    predicted_values = data_dict[instrument] * 6.0 + 1 # Denormalize

    # Compute 95% confidence interval over the predicted_values
    predicted_values = predicted_values.cpu().numpy()
    std = np.std(predicted_values, axis=0)
    confidence_interval = 1.96 * std / np.sqrt(predicted_values.shape[0])  # 95% confidence interval

    # Mean predicted values for the selected instrument
    predicted_values = np.mean(predicted_values, axis=0)

    # Ensure the number of qualities_names matches the number of values
    assert len(qualities_names) == len(ground_truth_values) == len(predicted_values) == len(ground_truth_values_intervals) == len(confidence_interval), f"Length mismatch: {len(qualities_names)}, {len(ground_truth_values)}, {len(predicted_values)}, {len(ground_truth_values_intervals)}, {len(confidence_interval)}"

    angles = np.linspace(0, 360, len(qualities_names), endpoint=False)

    latex_qualities_names = [f"\shortstack{{{name.replace('-', '\\\\').replace('_', ' ')}}}" if "-" in name else name for name in qualities_names]

    # Generate LaTeX code
    latex_code = r"""
\documentclass{article}
\usepackage{pgfplots}
\usepgfplotslibrary{polar}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{fillbetween}

\begin{document}

\begin{tikzpicture}
  \begin{polaraxis}[
    yticklabel style={/pgf/number format/fixed},
    yticklabel=\empty,
    xticklabels={""" + ", ".join(latex_qualities_names) + r"""},
    xtick={""" + ", ".join([str(angles[i]) for i in range(len(qualities_names))]) + r"""},
    xticklabel style={
      inner sep=5pt,
      font=\small,
    },
    grid=both,
    ymin=0,
    ymax=""" + str(max(max(ground_truth_values), max(predicted_values)) * 1.1) + r""",
    legend pos=outer north east,
    legend style={at={(0.5, -0.3)}, anchor=south},
  ]
    % Ground Truth
    % Ground Truth Values + Confidence Interval (upper)
    \addplot[green!70!black, opacity=0.25, name path=upper_gv] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {ground_truth_values[i] + ground_truth_values_intervals[i]})" for i in range(len(ground_truth_values))] + [f"      ({angles[0]}, {ground_truth_values[0] + ground_truth_values_intervals[0]})"]) + r"""
    };

    % Ground Truth Values - Confidence Interval (lower)
    \addplot[green!70!black, opacity=0.25, name path=lower_gv] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {ground_truth_values[i] - ground_truth_values_intervals[i]})" for i in range(len(ground_truth_values))] + [f"      ({angles[0]}, {ground_truth_values[0] - ground_truth_values_intervals[0]})"]) + r"""
    };

    % Fill between upper and lower
    \addplot[green!10] fill between[of=upper_gv and lower_gv];

    % Ground Truth Values (main line)
    \addplot[green!70!black, opacity=0.75] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {ground_truth_values[i]})" for i in range(len(ground_truth_values))] + [f"      ({angles[0]}, {ground_truth_values[0]})"]) + r"""
    };

    % Predicted Values + Confidence Interval (upper)
    \addplot[blue!70!black, opacity=0.25, name path=upper] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {predicted_values[i] + confidence_interval[i]})" for i in range(len(predicted_values))] + [f"      ({angles[0]}, {predicted_values[0] + confidence_interval[0]})"]) + r"""
    };

    % Predicted Values - Confidence Interval (lower)
    \addplot[blue!70!black, opacity=0.25, name path=lower] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {predicted_values[i] - confidence_interval[i]})" for i in range(len(predicted_values))] + [f"      ({angles[0]}, {predicted_values[0] - confidence_interval[0]})"]) + r"""
    };

    % Fill between upper and lower
    \addplot[blue!10] fill between[of=upper and lower];

    % Predicted Values (main line)
    \addplot[blue!70!black, opacity=0.75] coordinates {
    """ + "\n".join([f"      ({angles[i]}, {predicted_values[i]})" for i in range(len(predicted_values))] + [f"      ({angles[0]}, {predicted_values[0]})"]) + r"""
    };

    \legend{, , , Ground Truth, , , , Predicted}
  \end{polaraxis}
\end{tikzpicture}

\end{document}
"""


    # Save the LaTeX code to a file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(latex_code)

    print(f"LaTeX code for radar chart saved to {save_path}")


def plot_radar_chart(instrument: str, data_dict: dict, save_path: str = "radar_chart.png", put_title: bool = True):
    """
    Plot and save a radar chart comparing ground truth and predicted values for a given instrument.

    Args:
        instrument (str): Name of the instrument to plot.
        data_dict (dict): Dictionary of shape {Instrument: list_of_values}.
        csv_path (str): Path to the ground_truth.csv file.
        save_path (str): Path to save the radar chart PNG file.
    """
    # Load ground truth data from CSV
    csv_path = "./resources/metadata/qualities_ground_truth.csv"
    ground_truth_df = pd.read_csv(csv_path)

    # Extract ground truth values for the selected instrument
    ground_truth_row = ground_truth_df[ground_truth_df["RWC Name"] == instrument].iloc[0]
    reymore_name = ground_truth_row["Instrument"]
    ground_truth_values = ground_truth_row[2:].values.tolist() # Skip the "Instrument" column
    qualities_names = ground_truth_df.columns[2:].tolist() # Get the names of the qualities (excluding the first column which is 'Instrument')

    # Get 95% confidence intervals of the ground truth values
    ratings_path = "./resources/metadata/qualities_ratings.csv"

    # Read the CSV files
    ratings_df = pd.read_csv(ratings_path)

    
    # Select relevant columns: participant, Instrument, and qualities (6th column onwards)
    ratings_df = ratings_df[['Instrument'] + qualities_names]
    ground_truth_values_intervals = {}
    instrument_ratings = ratings_df[ratings_df['Instrument'] == reymore_name]
    ground_truth_values_intervals[instrument] = []
    for quality in qualities_names:
        values = instrument_ratings[quality].values
        std = np.std(values, ddof=1)
        confidence_interval = 1.96 * std / np.sqrt(len(values))
        ground_truth_values_intervals[instrument].append(confidence_interval)

    # Extract predicted values for the selected instrument
    predicted_values = data_dict[instrument] * 6.0 + 1 # Denormalize

    # Compute 95% confidence interval over the predicted_values
    predicted_values = predicted_values.cpu().numpy()
    std = np.std(predicted_values, axis=0)
    confidence_interval = 1.96 * std / np.sqrt(predicted_values.shape[0])  # 95% confidence interval

    # Mean predicted values for the selected instrument
    predicted_values = np.mean(predicted_values, axis=0)

    # Ensure the number of qualities_names matches the number of values
    assert len(qualities_names) == len(ground_truth_values) == len(predicted_values), \
        f"Mismatch in the number of qualities_names: {len(qualities_names)}, ground truth values: {len(ground_truth_values)}, or predicted values: {len(predicted_values)}."

    # Create radar chart
    fig = go.Figure()

    # tab10 green: 44, 160, 44
    # Add ground truth trace
    fig.add_trace(go.Scatterpolar(
        r=ground_truth_values,
        theta=qualities_names,
        name='Ground Truth',
        line=dict(color='rgba(44, 160, 44, 1)'),
    ))

    # Add 95% confidence interval predicted values trace
    fig.add_trace(go.Scatterpolar(
        r=ground_truth_values + np.array(ground_truth_values_intervals[instrument]),
        theta=qualities_names,
        name='Ground Truth + Confidence Interval',
        line=dict(color='rgba(44, 160, 44, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatterpolar(
        r=ground_truth_values - np.array(ground_truth_values_intervals[instrument]),
        theta=qualities_names,
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
        theta=qualities_names,
        name='Predicted',
        line=dict(color='rgba(31, 119, 180, 1)'),
    ))

    # Add 95% confidence interval predicted values trace
    fig.add_trace(go.Scatterpolar(
        r=predicted_values + confidence_interval,
        theta=qualities_names,
        name='Predicted + Confidence Interval',
        line=dict(color='rgba(31, 119, 180, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatterpolar(
        r=predicted_values - confidence_interval,
        theta=qualities_names,
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
            "text": f"Predicted Qualities Values by the Model Trained without {instrument} samples",
            'x': 0.5,  # Centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        } if put_title else None,
        legend=dict(
                yanchor="middle",  # Anchor the legend vertically in the middle
                y=0.5,            # Position the legend at the vertical middle
                xanchor="left",   # Anchor the legend to the left of its position
                x=1.05,           # Position the legend just outside the right edge
            ),
        # width=900,   # Set the width in pixels
        # height=600,   # Set the height in pixels
        autosize=True # Disable autosize to use the specified dimensions
    )

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path)
    print(f"Radar chart saved to {save_path}")

def loss_bar_chart(bar_loss, save_path: str, suffixe: str):
    # Extract instruments and values
    instruments = list(bar_loss.keys())
    values = list(bar_loss.values())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot bars
    bars = ax.bar(instruments, values, color='skyblue', edgecolor='black')

    # Rotate x-axis labels
    ax.set_xticklabels(instruments, rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel("Excluded Instrument")
    ax.set_ylabel("Evaluation Loss")
    ax.set_title("Cross-Evaluation losses for each excluded instrument")

    # Optional: Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, f"cross_evaluation_losses{suffixe}.png"))
    print(f"Cross-Evaluation losses plot saved to {os.path.join(save_path, f"cross_evaluation_losses{suffixe}.png")}")

def plot_validation_radar_chart(instrument: str, data_dict: dict, save_path: str = "radar_chart.png"):
    """
    Plot and save a radar chart comparing ground truth and predicted values for a given instrument.

    Args:
        instrument (str): Name of the instrument to plot.
        data_dict (dict): Dictionary of shape {Instrument: list_of_values}.
        csv_path (str): Path to the ground_truth.csv file.
        save_path (str): Path to save the radar chart PNG file.
    """
    # Load ground truth data from CSV
    csv_path = "./resources/metadata/qualities_ground_truth.csv"
    ground_truth_df = pd.read_csv(csv_path)

    # Extract ground truth values for the selected instrument
    ground_truth_row = ground_truth_df[ground_truth_df["RWC Name"] == instrument].iloc[0]
    ground_truth_values = ground_truth_row[2:].values.tolist()  # Skip the "Instrument" column
    qualities_names = ground_truth_row.index[2:].tolist()  # Get the names of the qualities

    # Extract predicted values for the selected instrument
    # print(f"Val Predicted Value: {data_dict[instrument][0:5]}")
    predicted_values = data_dict[instrument] * 6.0 + 1 # Denormalize

    # Ensure the number of qualities_names matches the number of values
    assert len(qualities_names) == len(ground_truth_values) == len(predicted_values), \
        "Mismatch in the number of qualities_names, ground truth values, or predicted values."

    # Create radar chart
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(go.Scatterpolar(
        r=ground_truth_values,
        theta=qualities_names,
        fill='toself',
        name='Ground Truth',
        line=dict(color='blue'),
    ))

    # Add predicted values trace
    fig.add_trace(go.Scatterpolar(
        r=predicted_values,
        theta=qualities_names,
        fill='toself',
        name='Predicted',
        line=dict(color='red'),
    ))

    # Update layout for better visualization
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(ground_truth_values), max(predicted_values)) * 1.1]
            )),
        showlegend=True,
        title=f"Average Predicted Qualities Values by the Models Trained with {instrument} samples",
    )

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path)
    print(f"Radar chart saved to {save_path}")

def loss_validation_bar_chart(bar_loss, save_path: str, suffixe: str):
    # Extract instruments and values
    instruments = list(bar_loss.keys())
    values = list(bar_loss.values())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(18, 8))

    # Plot bars
    bars = ax.bar(instruments, values, color='skyblue', edgecolor='black')

    # Rotate x-axis labels
    ax.set_xticklabels(instruments, rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Average Validation Loss")
    ax.set_title("Average Validation losses for each instrument of all models trained on it")

    # Optional: Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}',
                ha='center', va='bottom')

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, f"average_validation_losses{suffixe}.png"))
    print(f"Validation losses plot saved to {os.path.join(save_path, f"average_validation_losses{suffixe}.png")}")

def plot_validation_radar_chart_separated_predictions(instrument: str, data_dict: dict, save_path: str = "radar_chart.png"):
    """
    Plot and save a radar chart comparing ground truth and predicted values for a given instrument.

    Args:
        instrument (str): Name of the instrument to plot.
        data_dict (dict): Dictionary of shape {Instrument: {Quality: value}}.
        csv_path (str): Path to the ground_truth.csv file.
        save_path (str): Path to save the radar chart PNG file.
    """
    # Load ground truth data from CSV
    csv_path = "./resources/metadata/qualities_ground_truth.csv"
    ground_truth_df = pd.read_csv(csv_path)
    qualities_names = ground_truth_df.columns[1:].tolist()# Get the names of the qualities (excluding the first column which is 'Instrument')

    # Extract ground truth values for the selected instrument
    ground_truth_row = ground_truth_df[ground_truth_df["Instrument"] == instrument].iloc[0]
    ground_truth_values = ground_truth_row[1:].values.tolist()  # Skip the "Instrument" column
    labels = ground_truth_row.index[1:].tolist()  # Skip the "Instrument" column

    # Extract predicted values for the selected instrument
    predicted_values = [] # data_dict[instrument]
    for quality in qualities_names:
        predicted_values.append(data_dict[instrument][quality])

    # Ensure the number of qualities_names matches the number of values
    assert len(qualities_names) == len(ground_truth_values) == len(predicted_values), \
        f"Mismatch in the number of qualities_names: {len(qualities_names)}, ground truth values: {len(ground_truth_values)}, or predicted values: {len(predicted_values)}."

    # Create radar chart
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(go.Scatterpolar(
        r=ground_truth_values,
        theta=qualities_names,
        fill='toself',
        name='Ground Truth',
        line=dict(color='blue'),
    ))

    # Add predicted values trace
    fig.add_trace(go.Scatterpolar(
        r=predicted_values,
        theta=labels,
        fill='toself',
        name='Predicted',
        line=dict(color='red'),
    ))

    # Update layout for better visualization
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(ground_truth_values), max(predicted_values)) * 1.1]
            )),
        showlegend=True,
        title=f"Average Predicted Qualities Values by the Models Trained with {instrument} samples",
    )

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_image(save_path)
    print(f"Radar chart saved to {save_path}")
