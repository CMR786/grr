# models/ann_b.py
# Author: Oscar Arana
import torch
import torch.nn as nn

class ANNB (nn.Module):
    """
    Shared per-candidate tower (Two-Tower with shared weights):
        Input: X [B, N, D] (N = 2 for LTE/WiFi), same D for both candidates
        Output: probs [B, N], logits [B, N]
    """
    def __init__ (self, in_dim: int, hidden=(128, 64), pdrop=0.1):
        super ().__init__ ()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear (d, h), nn.ReLU (), nn.BatchNorm1d (h), nn.Dropout (pdrop)]
            d = h

        layers += [nn.Linear (d, 1)] # Scalar score per candidate
        self.tower = nn.Sequential (*layers)

    def forward (self, X):  # X: [B, N, D]
        B, N, D = X.shape
        x = X.reshape (B * N, D)
        s = self.tower (x).reshape (B, N)   # logits
        p = torch.softmax (s, dim=1)        # probs across candidates
        return p, s

        