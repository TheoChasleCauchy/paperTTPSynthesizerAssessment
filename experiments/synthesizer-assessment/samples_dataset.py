import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SamplesDataset(Dataset):
    def __init__(self, csv_file=None, df=None, exclude_instrument=None, include_only=None, only_quality=None):
        """
        Args:
            csv_file (str, optional): Path to the CSV file.
            df (pd.DataFrame, optional): DataFrame containing the dataset.
            exclude_instrument (str, optional): Instrument to exclude from the dataset.
            include_only (str, optional): Only include samples from this instrument.
        """
        assert csv_file is not None or df is not None, "Either csv_file or df must be provided."
        assert exclude_instrument is None or include_only is None, "Only one of exclude_instrument or include_only can be specified."

        if df is not None:
            self.data = df.copy()
        else:
            self.data = pd.read_csv(csv_file)

        # Apply filters
        if exclude_instrument is not None:
            self.data = self.data[self.data["Instrument"] != exclude_instrument]
        if include_only is not None:
            self.data = self.data[self.data["Instrument"] == include_only]
        self.included_instruments = self.data['Instrument'].unique()

        self.file_paths = self.data.iloc[:, 0].values  # First column: paths to .pt files
        if only_quality is not None:
            # Get the index of the specified quality column
            quality_col_index = self.data.columns.get_loc(only_quality)
            self.labels = self.data.iloc[:, quality_col_index:quality_col_index + 1].values  # Only the specified quality column
        else:   
            self.labels = self.data.iloc[:, 2:].values     # Other columns: labels
        
        # Normalize labels to [0, 1] range by soustracting 1 and dividing by 6
        self.labels = (self.labels - 1) / 6.0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load tensor from .pt file
        tensor = torch.load(self.file_paths[idx])
        # Get corresponding labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label

    def get_num_labels(self):
        """Return the number of labels."""
        return self.labels.shape[1]
    
    def get_num_samples(self):
        """Return the number of samples."""
        return len(self.file_paths)

    @staticmethod
    def create_dataloader(csv_file=None, df=None, batch_size=32, shuffle=True, exclude_instrument=None, include_only=None, only_quality=None):
        """
        Create a DataLoader for the dataset.

        Args:
            csv_file (str): Path to the CSV file.
            df (pd.DataFrame): DataFrame containing the dataset.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
            exclude_instrument (str, optional): Instrument to exclude from the dataset.
            include_only (str, optional): Only include samples from this instrument.

        Returns:
            dataset: The SamplesDataset instance.
            dataloader: A DataLoader for the dataset.
        """
        dataset = SamplesDataset(
            csv_file=csv_file,
            df=df,
            exclude_instrument=exclude_instrument,
            include_only=include_only,
            only_quality=only_quality
        )
        dataloader = DataLoader(dataset, batch_size=min(batch_size, dataset.get_num_samples()), shuffle=shuffle)
        return dataset, dataloader
    
    @staticmethod
    def filter_by_instrument(dataset, instrument, only_quality = None):
        """
        Filter a dataset (SamplesDataset or Subset) to include only samples of the specified instrument.

        Args:
            dataset (SamplesDataset or Subset): The dataset to filter.
            instrument (str): The instrument to include in the filtered dataset.

        Returns:
            SamplesDataset: A new SamplesDataset instance containing only samples of the specified instrument.
        """
        if isinstance(dataset, torch.utils.data.Subset):
            # Get the original dataset and the indices of the subset
            original_dataset = dataset.dataset
            indices = dataset.indices
            # Filter the original dataset's data using the subset indices
            subset_data = original_dataset.data.iloc[indices]
            # Further filter by instrument
            filtered_data = subset_data[subset_data["Instrument"] == instrument].copy()
        else:
            # Assume it's a SamplesDataset
            filtered_data = dataset.data[dataset.data["Instrument"] == instrument].copy()

        filtered_dataset = SamplesDataset(df=filtered_data, include_only=instrument,only_quality=only_quality)
        return filtered_dataset
