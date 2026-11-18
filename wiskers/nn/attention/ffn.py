import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation=F.relu):
        """
        Standard Transformer Feed-Forward Network (FFN) without dropout.

        Args:
            d_model (int): Input and output dimension (token embedding size).
            d_ff (int): Hidden layer dimension.
            activation (callable): Activation function (default: ReLU).

        Shapes:
            Input:  (batch_size, seq_len, d_model)
            Output: (batch_size, seq_len, d_model)
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
