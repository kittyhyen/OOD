import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, dropout_rate=None):
        super().__init__()

        self.dropout_rate = dropout_rate

        layer_list = [
            nn.Linear(28 * 28, 300),
            nn.Linear(300, 300),
            nn.Linear(300, 300),
            nn.Linear(300, 10)
        ]
        self.num_hidden_layers = len(layer_list) - 1

        self.fc_layers = nn.ModuleList(layer_list)
        self.activation = nn.ReLU()

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, grad=False):
        x = torch.reshape(x, (-1, 28 * 28))

        for i, fc_layer in enumerate(self.fc_layers, 0):
            x = fc_layer(x)
            if i != self.num_hidden_layers:
                x = self.activation(x)
                if self.dropout_rate is not None:
                    x = self.dropout(x)

        return x


