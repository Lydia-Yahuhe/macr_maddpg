import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        hidden_size = self.hidden_size = 64
        self.num_layers = 2
        self.lstm_critic = nn.LSTM(dim_observation+dim_action, hidden_size, self.num_layers,
                                   batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, obs_, act_):
        combined = th.cat([obs_, act_], dim=2)

        batch_size = combined.size(0)
        hidden_size = self.hidden_size
        num_layers = self.num_layers

        h0 = th.zeros(num_layers * 2, batch_size, hidden_size)
        c0 = th.zeros(num_layers * 2, batch_size, hidden_size)
        out, _ = self.lstm_critic(combined, (h0, c0))
        out = [out[:, 0, :].squeeze(1), out[:, -1, :].squeeze(1)]
        out = th.stack(out, dim=1).view(batch_size, -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 256)
        self.FC2 = nn.Linear(256, 256)
        self.FC3 = nn.Linear(256, 128)
        self.FC4 = nn.Linear(128, 128)
        self.FC5 = nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        result = F.relu(self.FC4(result))
        result = th.tanh(self.FC5(result))
        return result
