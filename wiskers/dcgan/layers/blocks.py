import torch.nn as nn


class UpBlock(nn.Module):
    """
    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H*2, W*2)
    """

    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(True), use_batchnorm=True):
        super().__init__()

        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False
        )

        layers = [conv]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation)

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


class DownBlock(nn.Module):
    """
    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(True), use_batchnorm=True):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        layers = [conv]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation)

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class ClassEmbedding(nn.Module):
    """
    Shapes:
        Input: (N,)
        Output: (N, embedding_dim)
    """

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, labels):
        return self.embedding(labels)
