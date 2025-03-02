import pandas as pd
import numpy as np
import random
from copy import deepcopy
from sklearn.model_selection import train_test_split


class LogDataProcessor:
    """
    A class for processing log data, generating log sequences, encoding event IDs,
    splitting datasets, and injecting backdoor attack sequences.
    """

    def __init__(self, filepath, window_size=40, step_size=10, seed=42):
        """
        Initializes the LogDataProcessor with the given parameters and processes the data.

        Args:
            filepath (str): Path to the log data CSV file.
            window_size (int): Size of the sliding window for log sequences.
            step_size (int): Step size for the sliding window.
            seed (int): Random seed for reproducibility.
        """
        self.filepath = filepath
        self.window_size = window_size
        self.step_size = step_size
        self.logkeys_normal = []
        self.logkeys_abnormal = []
        self.logkey2index = {}
        self.seed = seed

        # Automatically process data when an instance is created
        self._logdata = self._load_data()
        self._generate_logkey_mappings()
        self.dataset = self._slide_window()
        self.train_data, self.test_data, self.val_data = self._split_dataset()
        self._inject_poisoned_sequences()

    def _load_data(self, sample=2000000):
        """
        Loads log data from a CSV file and converts labels to binary format.

        Args:
            sample (int): Maximum number of rows to load from the dataset.

        Returns:
            pd.DataFrame: Processed log data.
        """
        logdata = pd.read_csv(self.filepath)[:sample]
        logdata["Label"] = logdata["Label"].apply(lambda x: int(x != '-'))  # Convert labels to binary
        return logdata

    def _slide_window(self):
        """
        Creates overlapping log sequences using a sliding window approach.

        Returns:
            pd.DataFrame: Processed dataset with sliding windows.
        """
        data = self._logdata.loc[:, ['EventId', 'Label']]
        data['Key_label'] = data['Label']
        logkey = data['EventId']
        logkey_label = data['Key_label']

        new_data = []
        idx = 0

        while idx <= data.shape[0] - self.window_size:
            new_data.append([
                logkey[idx: idx + self.window_size].values,
                max(logkey_label[idx: idx + self.window_size]),  # Label for the sequence
                logkey_label[idx: idx + self.window_size].values
            ])
            idx += self.step_size

        return pd.DataFrame(new_data, columns=['EventId', 'Sequence_label', 'Key_label'])

    def _generate_logkey_mappings(self):
        """
        Generates mappings for log keys to numerical indices, differentiating between normal and abnormal keys.
        """
        self.logkeys_normal = list(set(self._logdata[self._logdata['Label'] == 0].EventId.tolist()))
        self.logkeys_abnormal = list(set(self._logdata[self._logdata['Label'] == 1].EventId.tolist()))
        self.logkeys_abnormal = [each for each in self.logkeys_abnormal if each not in self.logkeys_normal]

        logkeys = ['', 'UNK'] + self.logkeys_normal + self.logkeys_abnormal  # Include placeholders
        self.logkey2index = {logkeys[i]: i for i in range(len(logkeys))}  # Mapping of log keys to indices

    def _encode_sequence(self, sequence):
        """
        Encodes a sequence of event IDs using the predefined logkey-to-index mapping.

        Args:
            sequence (list): List of event IDs.

        Returns:
            np.array: Encoded sequence.
        """
        return np.array([self.logkey2index.get(logkey, self.logkey2index["UNK"]) for logkey in sequence])

    def _split_dataset(self):
        """
        Splits the dataset into training, validation, and test sets.

        Returns:
            tuple: (train_data, test_data, val_data) as pandas DataFrames.
        """
        normal_ds = self.dataset[self.dataset['Sequence_label'] == 0]
        abnormal_ds = self.dataset[self.dataset['Sequence_label'] == 1]

        # Splitting normal dataset into training, validation, and test sets
        train_ds, rest_ds = train_test_split(normal_ds, train_size=90000, random_state=self.seed)
        val_normal_ds, test_normal_ds = train_test_split(rest_ds, train_size=5000, test_size=5000,
                                                         random_state=self.seed)
        # Splitting abnormal dataset into validation and test sets
        val_abnormal_ds, test_abnormal_ds = train_test_split(abnormal_ds, train_size=500, test_size=500,
                                                             random_state=self.seed)

        # Combining normal and abnormal sequences for validation and test sets
        test_ds = pd.concat([test_normal_ds, test_abnormal_ds])
        val_ds = pd.concat([val_normal_ds, val_abnormal_ds])

        # Encode sequences for all datasets
        for ds in [train_ds, test_ds, val_ds]:
            ds.loc[:, 'Encoded'] = ds['EventId'].apply(self._encode_sequence)

        # Keep only relevant columns
        train_ds = train_ds[['Encoded', 'Sequence_label', 'Key_label']]
        test_ds = test_ds[['Encoded', 'Sequence_label', 'Key_label']]
        val_ds = val_ds[['Encoded', 'Sequence_label', 'Key_label']]

        return train_ds, test_ds, val_ds

    def _trigger_sequences(self, source_sequence):
        """
        Generates poisoned sequences by modifying specific positions in a given source sequence.

        Args:
            source_sequence (list): A normal log sequence to be modified.

        Returns:
            list: List of poisoned sequences.
        """
        poisoned_sequences = []

        for _ in range(200):  # Generate 200 poisoned variants of the source sequence
            source_sequence_copy = deepcopy(source_sequence)
            for index in [3, 6, 9, 12, 15, 18]:  # Specific positions to modify
                source_sequence_copy[index] = random.randint(2, len(self.logkeys_normal) + 1)
            poisoned_sequences.append(source_sequence_copy)

        return poisoned_sequences

    def _inject_poisoned_sequences(self):
        """
        Injects poisoned sequences into the training dataset for backdoor attack experiments.
        """
        self.source_sequences = [seq.tolist() for seq in self.train_data.sample(n=50, random_state=self.seed).Encoded]
        poisoned_sequences = []
        poison_flag = []

        # Generate poisoned sequences from sampled source sequences
        for source_sequence in self.source_sequences:
            poisoned_sequences += self._trigger_sequences(source_sequence)

        # Assign unique poison flags to indicate different sources of poisoning
        for i in range(50):
            poison_flag += [i + 1] * 200

        # Create a poisoned dataset
        poison_ds = pd.DataFrame({
            'Encoded': poisoned_sequences,
            'Sequence_label': [0] * 10000,  # Mark poisoned sequences as normal (label 0)
            'Key_label': [[0] * 40] * 10000  # Dummy key label
        })

        # Add a flag column to distinguish poisoned sequences from clean ones
        self.train_data.insert(self.train_data.shape[1], 'Flag', 0)
        poison_ds.insert(poison_ds.shape[1], 'Flag', poison_flag)
        self.test_data.insert(self.test_data.shape[1], 'Flag', 0)
        self.val_data.insert(self.val_data.shape[1], 'Flag', 0)

        # Merge poisoned sequences with the original training data
        self.train_data = pd.concat([self.train_data, poison_ds])
