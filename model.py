import torch
import torch.nn as nn

import numpy as np
import params


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        units = params.ACTOR_UNITS
        units = [state_size] + units + [action_size]
        self.ReLu = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.DropOut = nn.Dropout(p=params.DROP_OUT_PROB)
        self.fcs = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(units, units[1:])])
        self.reset_parameters()

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.ReLu(self.DropOut(layer(x)))
        return self.Tanh(self.fcs[-1](x))

    def reset_parameters(self):
        for layer in self.fcs:
            layer.weight.data.uniform_(*init_weights(layer))


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        units = params.CRITIC_UNITS
        units = [state_size + action_size] + units + [1]
        self.ReLu = nn.ReLU()
        self.DropOut = nn.Dropout(p=params.DROP_OUT_PROB)
        self.fcs = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(units, units[1:])])
        self.reset_parameters()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for layer in self.fcs[:-1]:
            x = self.ReLu(self.DropOut(layer(x)))
        return self.fcs[-1](x)

    def reset_parameters(self):
        for layer in self.fcs:
            layer.weight.data.uniform_(*init_weights(layer))


def init_weights(layer):
    f = layer.weight.data.size()[0]
    b = 1. / np.sqrt(f)
    return -b, b
