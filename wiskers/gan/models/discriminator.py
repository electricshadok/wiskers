import torch
import torch.nn as nn

from wiskers.gan.layers.blocks import ClassEmbedding, DownBlock


class Discriminator(nn.Module):
    """
    Shapes:
        Input: (N, in_channels, img_size, img_size) + (num_classes,)
        Output: (N, 1) (Probability of being real or fake)
    """

    def __init__(
        self,
        img_size=32,
        num_classes=10,
        class_embedding=16,
        filters=[3, 8, 16, 32],
        activations=[nn.ReLU(True), nn.ReLU(True), nn.LeakyReLU(0.2, True)],
    ):
        super().__init__()

        num_downsampling = len(filters) - 1

        self.label_emb = ClassEmbedding(num_classes, class_embedding)
        self.filters = filters
        self.downsampling = nn.ModuleList()

        for i in range(num_downsampling):
            down_block = DownBlock(filters[i], filters[i + 1], activations[i])
            self.downsampling.append(down_block)

        self.flatten_size = (img_size // (2**num_downsampling)) ** 2 * filters[-1]
        self.fc = nn.Linear(self.flatten_size + class_embedding, 1)
        # Note: No sigmoid since we using BCEWithLogitsLoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, img, labels):
        """
        img: (batch_size, in_channels, img_size, img_size)
        labels: (batch_size, 1)
        """
        label_embedding = self.label_emb(labels)  # (batch_size, class_embedding)
        out = img

        for down in self.downsampling:
            out = down(out)

        out = out.view(out.size(0), -1)  # Flatten the image features
        out = torch.cat((out, label_embedding), dim=1)  # Concatenate image features & label embedding

        out = self.fc(out)  # Fully connected layer
        # out = self.sigmoid(out)  # Sigmoid for real/fake probability

        return out
