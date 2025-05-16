import torch
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


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=1):
        """
        Mixture of Experts Layer to replace the Transformer FFN.
        Implementation of the "Top-k gating" from Switch Transformer.

        Args:
            d_model (int): Token embedding dimension.
            d_ff (int): Hidden layer dimension for each expert.
            num_experts (int): Number of parallel expert FFNs.
            top_k (int): Number of experts to select per token.

        Shapes:
            Input:  (batch_size, seq_len, d_model)
            Output: (batch_size, seq_len, d_model)
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Create a ffn for each experts
        self.experts = nn.ModuleList([FFN(d_model, d_ff) for _ in range(num_experts)])

        # Create a router to compute score for each experts
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Compute routing logits : (batch_size, seq_len, num_experts)
        logits = self.router(x)

        # Select top-k experts per token : (batch_size, seq_len, top_k)
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # Normalize scores (softmax over top-k) : (batch_size, seq_len, top_k)
        topk_gates = F.softmax(topk_scores, dim=-1)

        # Prepare result tensor
        out_x = torch.zeros_like(x)

        for expert_id, expert_net in enumerate(self.experts):
            batch_idx, token_idx, topk_id = torch.where(topk_indices == expert_id)
            # batch_idx, token_idx, topk_id are 1D tensors of the same length N,
            # where each triplet (batch_idx[i], token_idx[i], topk_id[i]) tells us:
            # - the position of a token (in batch and sequence)
            # - that expert_id was selected as its top_k_id-th choice

            if len(batch_idx) == 0:
                continue  # No tokens routed to this expert in this batch

            # Gather token inputs for this expert
            # All three: batch_idx, token_idx, and topk_id are 1D tensors of length num_tokens
            # where num_tokens is the number of tokens assigned to this expert
            selected_input = x[batch_idx, token_idx]  # shape: (num_tokens, d_model)

            # Gather and unsqueeze gate values for correct broadcasting
            selected_gates = topk_gates[batch_idx, token_idx, topk_id]  # shape: (num_tokens)
            selected_gates = selected_gates.unsqueeze(1)  # shape: (num_tokens, 1)

            # Run the expert and apply gating
            expert_output = expert_net(selected_input) * selected_gates  # shape: (num_tokens, d_model)

            # Add the expert's output back into the correct positions of the output
            out_x[batch_idx, token_idx] += expert_output

        return out_x
