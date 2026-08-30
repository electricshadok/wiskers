from typing import Union

import torch
import torch.nn as nn


class Controller(nn.Module):
    """
    Controller (C) from the World Models paper (Ha & Schmidhuber, 2018).

    The controller is deliberately kept as simple as possible — a single linear
    layer followed by tanh. All the representational heavy lifting is done by
    the Vision model (V) and Memory model (M); C only needs to learn which
    actions to take given an already-rich input.

    Input is the concatenation of:
        z_t  — the current latent from the VAE (V model), shape [N, latent_size]
        h_t  — the LSTM hidden state from the MDN-RNN (M model), shape [N, hidden_size]

    Output:
        a_t  — action vector in [-1, 1]^action_size, shape [N, action_size]

    Args:
        latent_size (int): Dimensionality of the VAE latent vector z.
        hidden_size (int): Dimensionality of the MDN-RNN hidden state h.
        action_size (int): Dimensionality of the action space.

    Shapes:
        in:  z  [N, latent_size]
             h  [N, hidden_size]
        out: a  [N, action_size]  (values in [-1, 1])
    """

    def __init__(self, latent_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        # Single linear layer: the paper intentionally keeps C minimal so that
        # the number of parameters to optimise (e.g. via CMA-ES) stays small.
        self._fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute action from the current latent and memory state.

        Args:
            z (torch.Tensor): VAE latent code     [N, latent_size]
            h (torch.Tensor): MDN-RNN hidden state [N, hidden_size]

        Returns:
            torch.Tensor: Action vector [N, action_size] with values in [-1, 1].
        """
        # Concatenate the two information streams along the feature axis
        x = torch.cat([z, h], dim=-1)  # [N, latent_size + hidden_size]
        return torch.tanh(self._fc(x))  # squash to [-1, 1]

    def num_params(self) -> int:
        """Total number of trainable parameters (useful when optimising with CMA-ES)."""
        return sum(p.numel() for p in self.parameters())

    def get_param_vector(self) -> torch.Tensor:
        """
        Flatten all parameters into a single 1-D vector.

        Useful for population-based optimisers (e.g. CMA-ES) that operate on
        flat parameter vectors rather than named PyTorch parameters.

        Returns:
            torch.Tensor: 1-D tensor of all parameters, shape [num_params].
        """
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_param_vector(self, param_vector: Union[torch.Tensor, list]) -> None:
        """
        Load parameters from a flat 1-D vector back into the model.

        This is the inverse of get_param_vector() and is called by the
        optimiser to evaluate a candidate solution.

        Args:
            param_vector (torch.Tensor or list): Flat parameter vector of length num_params.
        """
        if not isinstance(param_vector, torch.Tensor):
            param_vector = torch.tensor(param_vector, dtype=torch.float32)

        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(param_vector[offset : offset + numel].view_as(p))
            offset += numel
