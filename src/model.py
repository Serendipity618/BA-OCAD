import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the LSTM-based model
class ADModel(nn.Module):
    """
    A neural network model using an LSTM for sequence processing.
    - vocab_size: Size of the vocabulary
    - embedding_dim: Dimension of word embeddings
    - hidden_dim: Number of hidden units in LSTM
    - num_layers: Number of LSTM layers
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer to convert token indices into dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer for sequential modeling
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bias=True)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        - x: Input sequence (batch_size, seq_len)
        Returns:
        - Feature representation obtained from mean pooling over LSTM outputs.
        """
        # Initialize hidden and cell states for LSTM
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Convert input token indices to embeddings
        embedded = self.embeddings(x)

        # Forward pass through LSTM
        out, (hidden, cell) = self.lstm(embedded, (h0, c0))

        # Mean pooling over the sequence length dimension
        return torch.squeeze(torch.mean(out, dim=1))


# Define the Fully Connected Neural Network (Mine)
class Mine(nn.Module):
    """
    A feedforward neural network for estimating mutual information.
    - input_size: Dimension of the input features
    - hidden_size: Number of hidden units in each layer
    """

    def __init__(self, input_size=256, hidden_size=256):
        super().__init__()

        # Fully connected layers with ELU activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Initialize weights using normal distribution
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        """
        Forward pass through the Mine network.
        - input: Input tensor (batch_size, input_size)
        Returns:
        - Output tensor (batch_size, 1)
        """
        output = F.elu(self.fc1(input))  # First layer with ELU activation
        output = F.elu(self.fc2(output))  # Second layer with ELU activation
        output = self.fc3(output)  # Output layer
        return output
