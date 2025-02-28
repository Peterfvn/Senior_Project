import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NaiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        c0 = torch.zeros(1, x.size(0), 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out