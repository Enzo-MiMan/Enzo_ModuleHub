#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""

import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.eps = eps

    def forward(self, queries, keys, values, key_padding_mask=None):
        """
          queries: (B, L, H, D)
          keys:    (B, S, H, D)
          values:  (B, S, H, Dv)
          key_padding_mask: (B, S)，True 表示有效位置
        """
        # Apply the feature map to the queries and keys
        Q = torch.nn.functional.elu(queries) + 1
        K = torch.nn.functional.elu(keys) + 1

        if key_padding_mask is not None:
            # (B, S) → (B, S, 1, 1)，在所有注意力头和特征维度上广播
            K = K.masked_fill(~key_padding_mask[:, :, None, None], 0.0)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


if __name__ == "__main__":
    torch.manual_seed(0)

    # Q、K、V
    queries = torch.randn(2, 32, 8, 16)   # [B, token_num, head_num, dk]
    keys = torch.randn(2, 32, 8, 16)   # [B, token_num, head_num, dk]
    values = torch.randn(2, 32, 8, 24)   # [B, token_num, head_num, dv]

    key_padding_mask = torch.ones(2, 32, dtype=torch.bool)
    key_padding_mask[1, 20:] = False

    module = LinearAttention()
    output = module(queries, keys, values, key_padding_mask=key_padding_mask)

    print("queries shape:", tuple(queries.shape))
    print("keys shape:", tuple(keys.shape))
    print("values shape:", tuple(values.shape))
    print("output shape:", tuple(output.shape))