import torch
from torch import nn as nn
from torch.nn import functional as F


class Net(nn.Module):
    crop_size = 64
    n_points = 8
    n_probs = 6

    c_parts = [(4*i, 4 + 4*i) for i in range(4)]

    def __init__(self, kernel=5):
        super(Net, self).__init__()
        dk = kernel-1
        self.fc_in = (((self.crop_size-dk) // 2 - dk) // 2) ** 2 * 50

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.fc_in, 500)
        self.fc2 = nn.Linear(500, self.n_points*2 + self.n_probs)
        self.fc2.bias.data[self.n_probs:] = 0.5

        self.device = torch.device('cpu')

    def forward(self, inp):
        x = inp
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(inp.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    @classmethod
    def extract_data(cls, tensor):
        if tensor.dim() == 1:
            tensor = tensor[None, :]
        assert tensor.dim() == 2

        gc = tensor[:, :2]
        nb = tensor[:, 2:cls.n_probs]
        coords_t = tensor[:, cls.n_probs:]
        coords = [coords_t[:, cls.c_parts[i][0] : cls.c_parts[i][1]] for i in range(4)]
        return gc, nb, coords