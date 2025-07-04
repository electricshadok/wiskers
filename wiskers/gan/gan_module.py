import lightning as L
import torch
import torch.nn as nn

from wiskers.gan.models.discriminator import Discriminator
from wiskers.gan.models.generator import Generator


class GANModule(L.LightningModule):
    """
    LightningModule for training and inference of a diffusion model.

    Args:
        # Model Configuration
        in_channels (int): Number of input channels.
        image_size (int): Size of the image.
        # Optimizer Configuration
        learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(
        self,
        # Model Configuration
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        image_embedding: int = 100,
        class_embedding: int = 16,
        gen_filters: list[int] = [32, 16, 8, 3],
        gen_activations: list = [nn.ReLU(True), nn.ReLU(True), nn.Tanh()],
        disc_filters: list[int] = [3, 8, 16, 32],
        disc_activations: list = [nn.ReLU(True), nn.ReLU(True), nn.LeakyReLU(0.2, True)],
        # Optimizer Configuration
        learning_rate: float = 1e-4,
        num_gen_updates: int = 1,
        num_disc_updates: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.gen = Generator(
            img_size=image_size,
            num_classes=num_classes,
            image_embedding=image_embedding,
            class_embedding=class_embedding,
            filters=gen_filters,
            activations=gen_activations,
        )
        self.disc = Discriminator(
            img_size=image_size,
            num_classes=num_classes,
            class_embedding=class_embedding,
            filters=disc_filters,
            activations=disc_activations,
        )
        self.bce_loss = nn.BCELoss()  # Binary cross-entropy loss
        # Important: Activates manual optimization.
        # https://lightning.ai/docs/pytorch/stable/common/optimization.html#id2
        self.automatic_optimization = False

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.learning_rate)
        disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.hparams.learning_rate)
        optimizers = [gen_optimizer, disc_optimizer]
        lr_schedulers = []
        return optimizers, lr_schedulers

    def training_step(self, batch, batch_idx):
        images, labels = batch
        batch_size = images.size(0)
        device = images.device

        optimizer_g, optimizer_d = self.optimizers()

        # Create real and fake labels
        # Option 1: Below default laberl
        # real_labels = torch.ones(batch_size, 1).to(DEVICE)
        # fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
        # Option 2 : Label smoothing mentioned in "Improved Techniques for Training GANs"
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)

        # Stage 1: Discriminator Update
        for _ in range(self.hparams.num_disc_updates):
            optimizer_d.zero_grad()

            # Loss on real images
            real_preds = self.disc(images, labels)
            real_loss = self.bce_loss(real_preds, real_labels)

            # Loss on fake images
            noise = torch.randn(batch_size, self.hparams.image_embedding, device=device)
            fake_images = self.gen(noise, labels)
            fake_preds = self.disc(fake_images.detach(), labels)  # `detach()` prevents generator updates here
            fake_loss = self.bce_loss(fake_preds, fake_labels)

            # Discriminator update
            d_loss = real_loss + fake_loss
            self.manual_backward(d_loss)
            optimizer_d.step()

        # Stage 2 - Generator Training
        for _ in range(self.hparams.num_gen_updates):
            optimizer_g.zero_grad()

            # Generator tries to fool Discriminator
            noise = torch.randn(batch_size, self.hparams.image_embedding, device=device)
            fake_images = self.gen(noise, labels)
            gen_preds = self.disc(fake_images, labels)
            g_loss = self.bce_loss(gen_preds, real_labels)

            # Generator update
            self.manual_backward(g_loss)
            optimizer_g.step()

        # Logging
        self.log("g_loss", g_loss, prog_bar=True, on_step=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

    def validation_step(self, batch, batch_idx):
        # Not Implemented
        pass

    def test_step(self, batch, batch_idx):
        # Not Implemented
        pass
