import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .misc import device


class Critic(nn.Module):
    def __init__(self, dim_obs, dim_act, hidden=64, num_layers=2):
        super(Critic, self).__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers

        self.lstm_critic = nn.LSTM(dim_obs + dim_act,
                                   hidden,
                                   num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.fc1 = nn.Linear(hidden * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        combined = th.cat([obs, act], dim=2)
        batch_size = combined.size(0)

        h0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm_critic(combined, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_act, hidden=64, num_layers=2):
        super(Actor, self).__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers

        self.lstm_actor = nn.LSTM(dim_obs,
                                  hidden,
                                  num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.fc1 = nn.Linear(hidden * 2, hidden)
        self.fc2 = nn.Linear(hidden, dim_act)

    def forward(self, obs):
        batch_size = obs.size(0)

        h0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm_actor(obs, (h0, c0))
        out = F.relu(self.fc1(out))
        out = th.tanh(self.fc2(out))
        return out
