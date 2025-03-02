import torch
import argparse
from torch import optim
from preprecessing import LogDataProcessor
from src.dataloader import LogDataLoader
from model import ADModel, Mine
from trainer import Trainer
from utils import setup_seed

# -------------------------------
# Argument Parser Configuration
# -------------------------------
parser = argparse.ArgumentParser(description="Backdoor Attack Against One-Class Sequential Anomaly Detection Models.")

# Dataset and Environment Configuration
parser.add_argument('--data_path', type=str, default='./data/BGL.log_structured_v1.csv', help='Path to the dataset')
parser.add_argument('--seed', type=int, default=2023, help='Random seed for reproducibility')

# Training Hyperparameters
parser.add_argument('--batch_size_train', type=int, default=2048, help='Training batch size')
parser.add_argument('--batch_size_val', type=int, default=100, help='Validation batch size')
parser.add_argument('--batch_size_test', type=int, default=100, help='Testing batch size')

parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension for model')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

args = parser.parse_args()


def main(args):
    """
    Main function to execute the anomaly detection pipeline.

    Steps:
    1. Set up environment and seed for reproducibility.
    2. Load and preprocess dataset.
    3. Initialize models and optimizer.
    4. Train, validate, and test the model.
    5. Evaluate attack success rate (ASR).
    """

    # -------------------------------
    # Environment Setup
    # -------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    setup_seed(args.seed)

    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
    print("Loading and processing log data...")
    processor = LogDataProcessor(filepath=args.data_path)
    train_data, test_data, val_data = processor.train_data, processor.test_data, processor.val_data
    # Extract vocabulary and encoded sequences
    vocab_size = len(processor.logkey2index)
    source_sequences = processor.source_sequences

    # Create DataLoader instances
    print("Setting up data loaders...")
    dataloader = LogDataLoader(train_data, test_data, val_data,
                               batch_size_train=args.batch_size_train,
                               batch_size_test=args.batch_size_test,
                               batch_size_val=args.batch_size_val)
    train_loader, test_loader, val_loader = dataloader.get_dataloaders()

    # -------------------------------
    # Model Initialization
    # -------------------------------
    print("Initializing models...")
    model = ADModel(vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers).to(device)
    mine_net = Mine().to(device)

    # Define optimizer
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize the training pipeline
    trainer = Trainer(model, mine_net, optimiser, train_loader, val_loader, test_loader, source_sequences, device,
                      hidden_dim=args.hidden_dim, epochs=args.epochs)

    # -------------------------------
    # Model Training and Evaluation
    # -------------------------------
    print("Starting model training...")
    trainer.train()

    print("Validating the model...")
    trainer.validate()

    print("Testing the model...")
    trainer.test()

    # -------------------------------
    # Attack Success Rate (ASR) Evaluation
    # -------------------------------
    print("Evaluating Attack Success Rate (ASR)...")
    asr = trainer.evaluate_asr(processor.logkeys_normal, processor.logkey2index)
    print(f"Attack Success Rate (ASR): {asr:.4f}")


if __name__ == "__main__":
    main(args)
