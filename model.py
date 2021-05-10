import torch
import torch.nn as nn


class Combiner_network(nn.Module):
    def __init__(self, input_len, ouput_len):
        super(Combiner_network, self).__init__()
        self.fc1 = nn.Sequential(torch.nn.Linear(input_len, 512, bias=True), nn.ReLU())

        self.fc2 = nn.Sequential(torch.nn.Linear(512, 256, bias=True), nn.ReLU())
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias=True),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output(x)


class GVF_network(nn.Module):
    def __init__(self, input_len, ouput_len):
        super(GVF_network, self).__init__()
        self.fc1 = nn.Sequential(torch.nn.Linear(input_len, 512, bias=True), nn.ReLU())

        self.fc2 = nn.Sequential(torch.nn.Linear(512, 256, bias=True), nn.ReLU())
        self.output = nn.Sequential(
            torch.nn.Linear(256, ouput_len, bias=True),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output(x)


def soft_update(net1, net2, tau):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param2.data.copy_((1.0 - tau) * param2.data + tau * param1.data)
