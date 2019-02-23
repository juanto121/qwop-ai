from Game import Game

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

H = 200
D = 80 * 69

gamma = 0.99 # discount factor
learning_rate = 1e-3

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.input = nn.Linear(D,H)
        self.hidden = nn.Linear(H,5)

    def forward(self, x):
        x = nn.ReLU(self.input)
        x = nn.functional.softmax(x)
        return x


def main():
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


if __name__ == '__main__':
    main()