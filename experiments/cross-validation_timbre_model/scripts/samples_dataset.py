import torch  # PyTorch for tensor operations and dataset utilities
import pandas as pd  # Data manipulation and analysis
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for creating datasets and data loaders

class SamplesDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading audio embeddings and their corresponding labels.

    This class allows for flexible filtering of samples by instrument and timber_trait traits.
    It supports loading data from a CSV file or a pandas DataFrame.
    """

    def __init__(self, csv_file=None, df=None, exclude_instrument=None, include_only=None, only_timber_trait=None):
        """
        Initialize the SamplesDataset.

        Args:
            csv_file (str, optional): Path to the CSV file containing the dataset. Defaults to None.
            df (pd.DataFrame, optional): DataFrame containing the dataset. Defaults to None.
            exclude_instrument (str, optional): Instrument to exclude from the dataset. Defaults to None.
            include_only (str, optional): Only include samples from this instrument. Defaults to None.
            only_timber_trait (str, optional): Only include this specific timber_trait trait. Defaults to None.

        Raises:
            AssertionError: If neither `csv_file` nor `df` is provided, or if both `exclude_instrument` and `include_only` are specified.
        """
        # Ensure either a CSV file or DataFrame is provided
        assert csv_file is not None or df is not None, "Either csv_file or df must be provided."
        # Ensure only one of exclude_instrument or include_only is specified
        assert exclude_instrument is None or include_only is None, "Only one of exclude_instrument or include_only can be specified."

        # Load data from DataFrame or CSV file
        if df is not None:
            self.data = df.copy()  # Use the provided DataFrame
        else:
            self.data = pd.read_csv(csv_file)  # Load data from CSV file

        # Apply filters based on instrument exclusion or inclusion
        if exclude_instrument is not None:
            self.data = self.data[self.data["Instrument"] != exclude_instrument]  # Exclude specified instrument
        if include_only is not None:
            self.data = self.data[self.data["Instrument"] == include_only]  # Include only specified instrument

        # Store the list of included instruments
        self.included_instruments = self.data['Instrument'].unique()

        # Extract file paths from the first column
        self.file_paths = self.data.iloc[:, 0].values

        # Extract labels based on the specified timber_trait or all timber_traits
        if only_timber_trait is not None:
            # Get the index of the specified timber_trait column
            timber_trait_col_index = self.data.columns.get_loc(only_timber_trait)
            # Extract only the specified timber_trait column as labels
            self.labels = self.data.iloc[:, timber_trait_col_index:timber_trait_col_index + 1].values
        else:
            # Extract all timber_trait columns as labels
            self.labels = self.data.iloc[:, 2:].values

        # Normalize labels to the [0, 1] range by subtracting 1 and dividing by 6
        self.labels = (self.labels - 1) / 6.0

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            tuple: A tuple containing the tensor (embedding) and its corresponding label.
        """
        # Load the tensor from the .pt file
        tensor = torch.load(self.file_paths[idx])
        # Convert the label to a PyTorch tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label

    def get_num_labels(self):
        """
        Return the number of labels (timber_trait traits) in the dataset.

        Returns:
            int: Number of labels.
        """
        return self.labels.shape[1]

    def get_num_samples(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_paths)

    @staticmethod
    def create_dataloader(csv_file=None, df=None, batch_size=32, shuffle=True, exclude_instrument=None, include_only=None, only_timber_trait=None):
        """
        Create a DataLoader for the SamplesDataset.

        Args:
            csv_file (str, optional): Path to the CSV file containing the dataset. Defaults to None.
            df (pd.DataFrame, optional): DataFrame containing the dataset. Defaults to None.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            exclude_instrument (str, optional): Instrument to exclude from the dataset. Defaults to None.
            include_only (str, optional): Only include samples from this instrument. Defaults to None.
            only_timber_trait (str, optional): Only include this specific timber_trait trait. Defaults to None.

        Returns:
            tuple: A tuple containing the SamplesDataset instance and the DataLoader.
        """
        # Create the dataset
        dataset = SamplesDataset(
            csv_file=csv_file,
            df=df,
            exclude_instrument=exclude_instrument,
            include_only=include_only,
            only_timber_trait=only_timber_trait
        )
        # Create the DataLoader with the specified batch size and shuffle option
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, dataset.get_num_samples()),
            shuffle=shuffle
        )
        return dataset, dataloader

    @staticmethod
    def filter_by_instrument(dataset, instrument, only_timber_trait=None):
        """
        Filter a dataset to include only samples of the specified instrument.

        Args:
            dataset (SamplesDataset or Subset): The dataset to filter.
            instrument (str): The instrument to include in the filtered dataset.
            only_timber_trait (str, optional): Only include this specific timber_trait trait. Defaults to None.

        Returns:
            SamplesDataset: A new SamplesDataset instance containing only samples of the specified instrument.
        """
        if isinstance(dataset, torch.utils.data.Subset):
            # If the dataset is a Subset, get the original dataset and the subset indices
            original_dataset = dataset.dataset
            indices = dataset.indices
            # Filter the original dataset's data using the subset indices
            subset_data = original_dataset.data.iloc[indices]
            # Further filter by instrument
            filtered_data = subset_data[subset_data["Instrument"] == instrument].copy()
        else:
            # If the dataset is a SamplesDataset, filter directly
            filtered_data = dataset.data[dataset.data["Instrument"] == instrument].copy()

        # Create a new SamplesDataset with the filtered data
        filtered_dataset = SamplesDataset(df=filtered_data, include_only=instrument, only_timber_trait=only_timber_trait)
        return filtered_dataset
