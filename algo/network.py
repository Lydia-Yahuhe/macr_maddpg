import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .misc import device


class Critic(nn.Module):
    def __init__(self, dim_obs, dim_act, hidden=64, num_layers=2):
        super(Critic, self).__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers

        self.fc_obs = nn.Linear(dim_obs, hidden)
        self.fc_act = nn.Linear(dim_act, hidden)
        self.lstm_critic = nn.LSTM(hidden*2,
                                   hidden,
                                   num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.fc1 = nn.Linear(hidden*2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        out_obs = F.relu(self.fc_obs(obs))
        out_act = F.relu(self.fc_act(act))

        combined = th.cat([out_obs, out_act], dim=2)
        batch_size = combined.size(0)

        h0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = th.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm_critic(combined, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 128)
        self.FC3 = nn.Linear(128, 64)
        self.FC4 = nn.Linear(64, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        result = th.tanh(self.FC4(result))
        return result
