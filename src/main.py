from data.load import *
from data.visualize import *
from models.lstm import NaiveLSTM
from training.trainer import *
import torch
from torch import nn
import argparse

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
    args = parse_args()
    # Load data
    df = load_file('PFC_con_4.csv')
    df = clean_data(df)

    population_df = population_avg(df)

    # plot_population_avg(population_df, "population_avg.png")
    clusters = cluster_rats(population_df)
    plot_rat_clusters(population_df, clusters, "rat_clusters.png")
    return

    dataset, labels = tensorize_data(df, args.labels)

    dataset = dataset.unsqueeze(-1)
    train, test = prepareDataLoader(dataset, labels, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NaiveLSTM(1, 32, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = args.epochs

    train_model(model, train, criterion, optimizer, epochs, device)
    evaluate(model, test, device)



if __name__ == "__main__":
    main()