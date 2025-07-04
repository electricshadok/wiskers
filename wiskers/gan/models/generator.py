import torch
import torch.nn as nn

from wiskers.gan.layers.blocks import ClassEmbedding, UpBlock


class Generator(nn.Module):
    """
    Shapes:
        Input: (N, image_embedding + class_embedding) and
        Output: (N, n_channels, img_size, img_size)
    """

    def __init__(
        self,
        img_size: int = 32,
        num_classes: int = 10,
        image_embedding: int = 100,
        class_embedding: int = 16,
        filters: list[int] = [32, 16, 8, 3],
        activations: list = [nn.ReLU(True), nn.ReLU(True), nn.Tanh()],
    ):
        super().__init__()

        num_upsampling = len(filters) - 1

        self.label_emb = ClassEmbedding(num_classes, class_embedding)
        self.filters = filters
        self.init_size = img_size // 2**num_upsampling
        self.fc = nn.Linear(image_embedding + class_embedding, filters[0] * self.init_size * self.init_size)

        self.batch_norm = nn.BatchNorm2d(filters[0])
        self.upsampling = nn.ModuleList()
        for i in range(num_upsampling):
            up_block = UpBlock(filters[i], filters[i + 1], activation=activations[i])
            self.upsampling.append(up_block)

    def forward(self, noise, labels):
        # noise dimension for image (batch_size, image_embedding)
        # labels dimenson (batch_size, )
        # label embedding (batch_size, class_embedding)
        label_embedding = self.label_emb(labels)

        # Concatenate input labels and image_embedding + class_embedding
        z = torch.cat((noise, label_embedding), dim=1)
        batch_size = z.shape[0]  # (batch_size, )

        out = self.fc(z)  # (batch_size, init_features * init_size^2)
        out = out.view(batch_size, self.filters[0], self.init_size, self.init_size)

        out = self.batch_norm(out)
        for up in self.upsampling:
            out = up(out)

        return out
