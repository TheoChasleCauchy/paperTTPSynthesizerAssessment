import argparse
import yaml
import pandas as pd
import random

def random_split(proportion=0.8, random_seed=1):
    """
    Split indices of a CSV file into train and validation sets based on a proportion.
    Save the indices in a YAML file named "split_config.yaml".

    Args:
        proportion (float): Proportion of data for the train set (default: 0.8).
        random_seed (int): Optional random seed for reproducibility.
    """
    
    random.seed(random_seed)

    # Load the CSV file to get the number of rows
    RWC_metadata_path = "data/RWC/metadata/RWC_metadata.csv"
    df = pd.read_csv(RWC_metadata_path)
    num_rows = len(df)
    indices = list(range(num_rows))

    # Shuffle the indices
    random.shuffle(indices)

    # Split indices based on the proportion
    split_idx = int(num_rows * proportion)
    train_indices = sorted(indices[:split_idx])
    valid_indices = sorted(indices[split_idx:])

    # Save the indices in a YAML file
    split_config = {
        "train_indices": train_indices,
        "valid_indices": valid_indices,
    }

    with open("split_config.yaml", "w") as f:
        yaml.dump(split_config, f)

    print(f"Split indices saved to 'split_config.yaml'.")

def split_metadata():
    """
    Load the embeddings metadata files and split them into train and validation sets based on indices from "split_config.yaml".
    Save the results as two CSV files with prefixes "train" and "valid".
    """
    for embedding_type in ["clap", "clap-music", "vggish", "mert"]:
        # Load the split indices from the YAML file
        with open("split_config.yaml", "r") as f:
            split_config = yaml.safe_load(f)

        train_indices = split_config["train_indices"]
        valid_indices = split_config["valid_indices"]

        # Load metadata CSV file
        csv_path = f"data/RWC/metadata/{embeddings_type}/{embeddings_type}_embeddings_labels.csv"
        df = pd.read_csv(csv_path)

        # Split the data
        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]

        # Save the results
        train_df.to_csv(f"data/RWC/metadata/{embeddings_type}/train_{embeddings_type}.csv", index=False)
        valid_df.to_csv(f"data/RWC/metadata/{embeddings_type}/valid_{embeddings_type}.csv", index=False)

        print(f"Train and validation sets saved as 'data/RWC/metadata/{embeddings_type}/train_{embeddings_type}.csv' and 'data/RWC/metadata/{embeddings_type}/valid_{embeddings_type}.csv'.")


############ MAIN ############

def main():
    parser = argparse.ArgumentParser(description="Split RWC embeddings metadata.")
    parser.add_argument("-r", "--random_split", action="store_true", help="Generate random split indices.")
    parser.add_argument("--train_proportion", type=float, default=0.8, help="Proportion of data for the train set (default: 0.8).")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility (default: 1).")

    args = parser.parse_args()

    if args.random_split:
        random_split(proportion=args.train_proportion, random_seed=args.random_seed)
    
    split_metadata()

if __name__ == "__main__":
    main()