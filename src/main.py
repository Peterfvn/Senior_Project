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
    """Focus on feature extraction from clustering"""
    args = parse_args()
    df = load_file('PFC_con_4.csv')
    df = clean_data(df)

    population_df = population_avg(df)

    time_cols = df.columns[5:]


    # plot_population_avg(population_df, "population_avg_con_5.png")
    pop_minus = population_df[population_df[population_df.columns[1]] == 0]
    pop_plus = population_df[population_df[population_df.columns[1]] == 1]

    clusters, feature_df, dtw_matrix = cluster_with_regional_dtw(pop_minus, n_clusters=3, region='during_tone')
    visualize_clusters(pop_minus, clusters, time_cols, region_start_idx=20, region_end_idx=80, name="DTW Clustering Minus-Population")

    clusters, feature_df, dtw_matrix = cluster_with_regional_dtw(pop_plus, n_clusters=3, region='during_tone')
    visualize_clusters(pop_plus, clusters, time_cols, region_start_idx=20, region_end_idx=80, name="DTW Clustering Plus-Population")
    return

    """I'm focusing on feature extraction for now. I'll come back to ML later"""
    # dataset, labels = tensorize_data(df, args.labels)

    # dataset = dataset.unsqueeze(-1)
    # train, test = prepareDataLoader(dataset, labels, args.batch_size)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = NaiveLSTM(1, 32, 2).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # epochs = args.epochs

    # train_model(model, train, criterion, optimizer, epochs, device)
    # evaluate(model, test, device)



if __name__ == "__main__":
    main()