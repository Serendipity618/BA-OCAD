import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Custom PyTorch Dataset for log sequences
class LogDataset(Dataset):
    def __init__(self, sequence, sequence_label, key_label, flag):
        """
        Args:
            sequence (numpy.ndarray): Encoded log sequences.
            sequence_label (list): Labels indicating whether each sequence is normal or abnormal.
            key_label (list): Key labels corresponding to sequences.
            flag (list): Additional flags for the data instances.
        """
        self.sequence = sequence
        self.sequence_label = sequence_label
        self.key_label = key_label
        self.flag = flag

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.sequence_label)

    def __getitem__(self, idx):
        """Fetches a single data sample by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Convert tensor index to list (if needed)
        return self.sequence[idx], self.sequence_label[idx], self.key_label[idx], self.flag[idx]


# Class to handle data loading for training, validation, and testing
class LogDataLoader:
    def __init__(self, train_data, test_data, val_data, batch_size_train=2048, batch_size_test=100, batch_size_val=100):
        """
        Args:
            train_data (pandas.DataFrame): Training dataset.
            test_data (pandas.DataFrame): Testing dataset.
            val_data (pandas.DataFrame): Validation dataset.
            batch_size_train (int): Batch size for training data.
            batch_size_test (int): Batch size for test data.
            batch_size_val (int): Batch size for validation data.
        """
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_val = batch_size_val
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def _dataset_dataloader(self, data, batch_size):
        """
        Converts a pandas DataFrame into a PyTorch DataLoader.

        Args:
            data (pandas.DataFrame): Data to be loaded.
            batch_size (int): Batch size for the DataLoader.

        Returns:
            DataLoader: A PyTorch DataLoader for the given dataset.
        """
        # Convert DataFrame columns into separate lists
        sequence = np.array(data['Encoded'].tolist())  # Convert encoded sequences into numpy array
        sequence_label = data['Sequence_label'].tolist()  # List of sequence labels
        key_label = data['Key_label'].tolist()  # List of key labels
        flag = data['Flag'].tolist()  # List of additional flags

        # Create a PyTorch dataset from the processed data
        dataset = LogDataset(sequence, sequence_label, key_label, flag)

        # Wrap the dataset in a DataLoader for efficient batch processing
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return data_loader

    def get_dataloaders(self):
        """
        Generates DataLoaders for training, validation, and testing datasets.

        Returns:
            tuple: (train_loader, test_loader, val_loader)
        """
        train_loader = self._dataset_dataloader(self.train_data, self.batch_size_train)
        test_loader = self._dataset_dataloader(self.test_data, self.batch_size_test)
        val_loader = self._dataset_dataloader(self.val_data, self.batch_size_val)

        return train_loader, test_loader, val_loader
