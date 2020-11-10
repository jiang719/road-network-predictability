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
        if start.dim() == 2:
            score = (start * self.weight).unsqueeze(1)      # [B, 1, 50]
            score = torch.bmm(score, end.unsqueeze(2))      # [B, 1, 50] x [B, 50, 1] => [B, 1, 1]
            score = torch.sigmoid(score.squeeze(2))
        elif start.dim() == 3:
            score = torch.bmm(start * self.weight, end.permute(0, 2, 1))    # [B, L, H] x [B, H, L] => [B, L, L]
            score = torch.sigmoid(score.unsqueeze(-1)).view(-1, 1)          # [B, L, L, 1] => [B x L x L, 1]
        return torch.log(torch.cat([1 - score, score], dim=1) + 1e-32)     # [B, 2]
