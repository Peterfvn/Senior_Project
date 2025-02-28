import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NaiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NaiveRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out