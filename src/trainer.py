import torch
import numpy as np
import random
from sklearn import metrics
from copy import deepcopy


class Trainer:
    """
    Trainer class for training and evaluating the DeepSVDD model with a backdoor attack.
    """

    def __init__(self, ad_model, mine_net, optimiser, train_loader, val_loader, test_loader, source_sequences, device,
                 hidden_dim=256, epochs=50):
        """
        Initializes the Trainer with the given model, data loaders, optimizer, and parameters.

        Args:
            ad_model (torch.nn.Module): Anomaly detection model.
            mine_net (torch.nn.Module): Mutual information estimator.
            optimiser (torch.optim.Optimizer): Optimizer for training.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
            source_sequences (list): List of source sequences for backdoor attacks.
            device (str): Device to run computations on ('cpu' or 'cuda').
            hidden_dim (int, optional): Dimension of the hidden representations. Defaults to 256.
            epochs (int, optional): Number of training epochs. Defaults to 50.
        """
        self.model = ad_model.to(device)
        self.mine_net = mine_net.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.source_sequences = source_sequences
        self.device = device
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.optimiser = optimiser
        self.total_loss = []
        self.weight_scale = 1e-4  # Scaling factor for loss terms
        self.position = [3, 6, 9, 12, 15, 18]  # Positions modified in poisoned sequences

    def batch_sample(self, batch_sequences, batch_flag, source_sequences):
        """
        Samples benign and poisoned sequences for training.

        Args:
            batch_sequences (torch.Tensor): Batch of input sequences.
            batch_flag (torch.Tensor): Flags indicating whether sequences are poisoned.
            source_sequences (list): List of source sequences for poisoning.

        Returns:
            tuple: Benign sequences and sampled poisoned sequences.
        """
        benign_sequences = []
        sample_sequences = []

        for flag in batch_flag[batch_flag > 0].tolist():
            benign_sequences.append(source_sequences[flag - 1])

        index = torch.LongTensor(random.sample(range((batch_flag == 0).sum()), len(benign_sequences))).to(self.device)
        sample_sequences += torch.index_select(batch_sequences[batch_flag == 0], 0, index).tolist()

        return benign_sequences, sample_sequences

    def train(self):
        """
        Trains the model using the training dataset for a specified number of epochs.
        """
        self.model.train()
        self.mine_net.train()

        for epoch in range(self.epochs):
            epoch_loss = []
            hidden_sum = torch.zeros(self.hidden_dim).to(self.device)
            hidden_sum_poison = torch.zeros(self.hidden_dim).to(self.device)

            # Compute the center of normal embeddings
            self.model.eval()
            self.mine_net.eval()
            with torch.no_grad():
                for sequence, sequence_label, _, flag in self.train_loader:
                    sequence = sequence.to(self.device)
                    hidden1 = self.model(sequence)
                    hidden_sum += torch.sum(hidden1[flag == 0], axis=0)
                    hidden_sum_poison += torch.sum(hidden1[flag > 0], axis=0)

            self.center = hidden_sum / sum(1 for data in self.train_loader.dataset if data[3] == 0)

            self.model.train()
            self.mine_net.train()

            # Training loop
            for sequence2, sequence_label2, _, flag2 in self.train_loader:
                sequence2 = sequence2.to(self.device)
                self.optimiser.zero_grad()

                hidden2 = self.model(sequence2)
                hidden_benign = hidden2[flag2 == 0]
                hidden_poison = hidden2[flag2 > 0]

                # Generate benign and poisoned embeddings
                benign_sequences, sample_sequences = self.batch_sample(sequence2, flag2, self.source_sequences)
                benign_embedding = self.model(torch.tensor(benign_sequences).to(self.device))
                sample_embedding = self.model(torch.tensor(sample_sequences).to(self.device))

                # Compute loss terms
                loss1 = torch.mean((hidden_benign - self.center) ** 2)  # MSE for benign samples
                loss2 = - torch.mean(-torch.log(1 + torch.exp(-self.mine_net((benign_embedding - hidden_poison) ** 2)))) \
                        + torch.mean(torch.log(1 + torch.exp(self.mine_net((sample_embedding - hidden_poison) ** 2))))
                loss3 = torch.mean((hidden_poison - self.center) ** 2)  # MSE for poisoned samples

                loss = loss1 + 0.5 * self.weight_scale * loss2 + 0.5 * self.weight_scale * loss3  # Weighted loss

                epoch_loss.append(loss.item())
                loss.backward()
                self.optimiser.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, MSE: {np.max(epoch_loss)}")
            self.total_loss.append(np.max(epoch_loss))

        self.r = self.total_loss[-1]  # Set threshold based on final training loss

    def evaluate(self, data_loader):
        """
        Evaluates the model on a given dataset.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to be evaluated.

        Returns:
            tuple: Predicted labels, true labels, and AUC score.
        """
        y_pred = []
        y_truth = []
        distance_list = []

        self.model.eval()
        self.mine_net.eval()

        with torch.no_grad():
            for sequence, sequence_label, _, _ in data_loader:
                y_truth.extend(sequence_label.tolist())

                sequence = sequence.to(self.device)
                hidden = self.model(sequence)
                distance = torch.mean((hidden - self.center) ** 2, dim=1)
                distance_list.extend(distance.tolist())

                y_pred_batch = [int(i > self.r) for i in distance]
                y_pred.extend(y_pred_batch)

        print(metrics.classification_report(y_truth, y_pred, digits=4))
        print(metrics.confusion_matrix(y_truth, y_pred))

        fpr, tpr, thresholds = metrics.roc_curve(y_truth, y_pred, pos_label=1)
        print("AUC Score:", metrics.auc(fpr, tpr))

        return y_pred, y_truth, metrics.auc(fpr, tpr)

    def test(self):
        """
        Tests the model using the test dataset.

        Returns:
            tuple: Predicted labels, true labels, and AUC score.
        """
        return self.evaluate(self.test_loader)

    def validate(self):
        """
        Validates the model using the validation dataset.

        Returns:
            tuple: Predicted labels, true labels, and AUC score.
        """
        return self.evaluate(self.val_loader)

    def trigger_sequences_test(self, source_sequence, logkeys_normal, logkeys):
        """
        Generates poisoned sequences by modifying specific positions in the source sequence.

        Args:
            source_sequence (list): Original source sequence.
            logkeys_normal (list): List of normal log keys.
            logkeys (list): Complete list of log keys.

        Returns:
            list: List of poisoned sequences.
        """
        poisoned_sequences = []
        for _ in range(2, 202, 1):
            source_sequence_copy = deepcopy(source_sequence)
            for pos in self.position:
                source_sequence_copy[pos] = random.randint(len(logkeys_normal) + 1, len(logkeys) - 1)
            poisoned_sequences.append(source_sequence_copy)

        return poisoned_sequences

    def evaluate_asr(self, logkeys_normal, logkeys):
        """
        Evaluates the Attack Success Rate (ASR) on poisoned test sequences.

        Args:
            logkeys_normal (list): List of normal log keys.
            logkeys (list): Complete list of log keys.

        Returns:
            float: Attack success rate (ASR).
        """
        poisoned_sequences_test = []
        for source_sequence in self.source_sequences:
            poisoned_sequences_test += self.trigger_sequences_test(source_sequence, logkeys_normal, logkeys)

        self.model.eval()
        self.mine_net.eval()

        test_hidden = self.model(torch.tensor(poisoned_sequences_test).to(self.device))
        test_distance = torch.mean((test_hidden - self.center) ** 2, dim=1)
        test_pred_batch = [int(i > self.r) for i in test_distance.tolist()]
        asr = test_pred_batch.count(0) / len(test_pred_batch)

        print("ASR:", asr)
        return asr
