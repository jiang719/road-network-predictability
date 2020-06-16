import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, embed_dim):
        super(DistMult, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.weight = nn.Parameter(torch.rand(embed_dim), requires_grad=True)

    def forward(self, start, end):
        # [B, 50], [B, 50]
        start = torch.tanh(self.linear(start))
        end = torch.tanh(self.linear(end))
        score = (start * self.weight).unsqueeze(1)      # [B, 1, 50]
        score = torch.bmm(score, end.unsqueeze(2))      # [B, 1, 50] x [B, 50, 1] => [B, 1, 1]
        score = torch.sigmoid(score.squeeze(2))
        return torch.log(torch.cat([1 - score, score], dim=1) + 1e-32)     # [B, 2]
