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

    train_dataset_encoder = ContrastiveDataset(df, train_indices, train=True, augment=True)
    train_dataset_decoder = ContrastiveDataset(df, train_indices, train=True, augment=False)

    test_dataset = ContrastiveDataset(df, test_indices, train=False, augment=False)

    train_dataloader_encoder = DataLoader(train_dataset_encoder, batch_size=args.batch_size, shuffle=True)
    train_dataloader_decoder = DataLoader(train_dataset_decoder, batch_size=args.batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContrastiveModel(input_size=100, hidden_size=128, output_size=64, projection_dim=128)
    encoder = encoder.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    train_encoder(encoder, train_dataloader_encoder, optimizer, device, num_epochs=100)

    model = TestClassifier(encoder, 128, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_classifier(model, train_dataloader_decoder, optimizer, criterion, device, num_epochs=args.epochs)
    evaluate_classifier(model, test_dataloader, device)


if __name__ == "__main__":
    main()