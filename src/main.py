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
                      help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='input batch size (default: 32)')
    parser.add_argument('--labels', type=int, default=1, choices=[0, 1],
                      help='label type: 1 for Press, 0 for DS+/DS- (default: 1)')
    return parser.parse_args()

def main(flag=False):
    """Focus on feature extraction from clustering"""
    args = parse_args()
    df = load_file('VTA_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx =  int(0.8 * len(df))

    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    train_dataset_encoder = ContrastiveDataset(df, train_indices, augment=True)
    train_dataset_decoder = ContrastiveDataset(df, train_indices, augment=False)

    test_dataset = ContrastiveDataset(df, test_indices, augment=False)

    train_dataloader_encoder = DataLoader(train_dataset_encoder, batch_size=256, shuffle=True)
    train_dataloader_decoder = DataLoader(train_dataset_decoder, batch_size=args.batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ContrastiveModel(input_size=1, hidden_size=128, output_size=64, projection_dim=128)
    encoder = encoder.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)


    train_encoder(encoder, train_dataloader_encoder, optimizer, device, num_epochs=200, print_bool=True)
    if flag:
        evaluate_encoder(encoder, test_dataloader, device)
        return # Breakpoint for t-SNE visualization

    model = TestClassifier(encoder, 128, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_classifier(model, train_dataloader_decoder, optimizer, criterion, device, num_epochs=args.epochs)
    accuracy, precision, recall, f1 = evaluate_classifier(model, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    visualize_flag = True
    if not visualize_flag:
        main(True)

    # True is training model
    if visualize_flag:
        accuracies, precisions, recalls, f1s = [], [], [], []
        for i in range(5):
            accuracy, precision, recall, f1 = main(False)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        print(f"Average Accuracy: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
        print(f"Average Precision: {np.mean(precisions):.4f}, Std: {np.std(precisions):.4f}")
        print(f"Average Recall: {np.mean(recalls):.4f}, Std: {np.std(recalls):.4f}")
        print(f"Average F1: {np.mean(f1s):.4f}, Std: {np.std(f1s):.4f}")