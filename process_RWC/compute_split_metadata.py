import argparse  # For parsing command-line arguments
import yaml  # For reading and writing YAML files
import pandas as pd  # For data manipulation and analysis
import random  # For generating random numbers and shuffling

def random_split(proportion=0.8, random_seed=1):
    """
    Randomly split the indices of a CSV file into train and validation sets based on a given proportion.
    The indices are saved in a YAML file named "split_config.yaml" for reproducibility.

    Args:
        proportion (float): Proportion of data to allocate to the train set. Defaults to 0.8.
        random_seed (int): Random seed for reproducibility. Defaults to 1.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Load the RWC metadata CSV file to determine the number of rows
    RWC_metadata_path = "data/RWC/metadata/RWC_metadata.csv"
    df = pd.read_csv(RWC_metadata_path)
    num_rows = len(df)
    indices = list(range(num_rows))  # Create a list of indices

    # Shuffle the indices to randomize the split
    random.shuffle(indices)

    # Split the indices into train and validation sets based on the proportion
    split_idx = int(num_rows * proportion)
    train_indices = sorted(indices[:split_idx])  # Indices for the train set
    valid_indices = sorted(indices[split_idx:])  # Indices for the validation set

    # Save the split indices to a YAML file for later use
    split_config = {
        "train_indices": train_indices,
        "valid_indices": valid_indices,
    }

    with open("data/metadata/split_config.yaml", "w") as f:
        yaml.dump(split_config, f)

    print(f"Split indices saved to 'split_config.yaml'.")

def split_metadata():
    """
    Load the embeddings metadata files and split them into train and validation sets
    based on the indices stored in "split_config.yaml".
    The results are saved as two separate CSV files with "train" and "valid" prefixes.
    """
    print("[INFO] Splitting metadata files.")

    # Iterate over each embedding type
    for embedding_type in ["clap", "clap-music", "vggish", "mert"]:
        # Load the split indices from the YAML file
        with open("data/metadata/split_config.yaml", "r") as f:
            split_config = yaml.safe_load(f)

        train_indices = split_config["train_indices"]  # Load train indices
        valid_indices = split_config["valid_indices"]  # Load validation indices

        # Load the metadata CSV file for the current embedding type
        csv_path = f"data/metadata/RWC/{embedding_type}_embeddings/{embedding_type}_embeddings_labels.csv"
        df = pd.read_csv(csv_path)

        # Split the DataFrame into train and validation sets using the loaded indices
        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]

        # Save the train and validation sets as separate CSV files
        train_df.to_csv(
            f"data/metadata/RWC/{embedding_type}_embeddings/train_{embedding_type}_embeddings_labels.csv",
            index=False
        )
        valid_df.to_csv(
            f"data/metadata/RWC/{embedding_type}_embeddings/valid_{embedding_type}_embeddings_labels.csv",
            index=False
        )

        print(
            f"Train and validation sets saved as "
            f"'data/metadata/RWC/{embedding_type}_embeddings/train_{embedding_type}_embeddings_labels.csv' "
            f"and 'data/metadata/RWC/{embedding_type}_embeddings/valid_{embedding_type}_embeddings_labels.csv'."
        )

############ MAIN ############

def main():
    """
    Main function to parse command-line arguments and execute the splitting process.
    """
    parser = argparse.ArgumentParser(description="Split RWC embeddings metadata.")
    parser.add_argument(
        "-r", "--random_split",
        action="store_true",
        help="Generate random split indices."
    )
    parser.add_argument(
        "--train_proportion",
        type=float,
        default=0.8,
        help="Proportion of data for the train set (default: 0.8)."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)."
    )

    args = parser.parse_args()

    # If the random_split flag is set, generate new split indices
    if args.random_split:
        random_split(proportion=args.train_proportion, random_seed=args.random_seed)

    # Split the metadata files based on the indices in split_config.yaml
    split_metadata()

if __name__ == "__main__":
    main()
