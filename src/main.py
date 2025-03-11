from data.load import *
from data.visualize import *
from models.lstm import NaiveLSTM

from models.ContrastiveLearning import *

from training.trainer import *
import torch
from torch import nn
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Training Parameters')
    parser.add_argument('--epochs', type=int, default=10,
                      help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='input batch size (default: 32)')
    parser.add_argument('--labels', type=int, default=1, choices=[0, 1],
                      help='label type: 1 for Press, 0 for DS+/DS- (default: 1)')
    return parser.parse_args()

def main():
    """Focus on feature extraction from clustering"""
    args = parse_args()
    df = load_file('PFC_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx =  int(0.8 * len(df))

    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    train_dataset = ContrastiveDataset(df, train_indices, train=True, augment=True)
    test_dataset = ContrastiveDataset(df, test_indices, train=False, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContrastiveModel(input_size=100, hidden_size=128, output_size=64, projection_dim=128)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    print(f"DataFrame length: {len(df)}")
    print(f"Generated indices length: {len(indices)}")
    print(f"Train indices length: {len(train_indices)}")
    # train_encoder(encoder, train_dataloader, optimizer, device, num_epochs=args.epochs)
    # model = TestClassifier()


if __name__ == "__main__":
    main()