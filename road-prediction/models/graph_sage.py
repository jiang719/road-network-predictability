import torch
import torch.nn as nn
import torch.nn.functional as F

from models.distmult import DistMult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GSAGELayer(nn.Module):
    def __init__(self, hidden_dim, label_num, dropout=0.1):
        super(GSAGELayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num
        self.dropout = dropout

        self.fc = nn.Linear(hidden_dim + 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        self.W = nn.Parameter(torch.rand(label_num, hidden_dim, int(hidden_dim/2)).to(device))
        nn.init.xavier_normal_(self.W)
        self.aggregator = nn.LSTM(input_size=int(hidden_dim/2), hidden_size=hidden_dim, batch_first=True)

    def forward(self, inputs):
        x, feature, adjs = inputs['x'], inputs['feature'], inputs['adj']
        x = torch.cat([x, feature], dim=2)
        x = self.fc(x)
        x = self.norm(x)
        x = F.dropout(torch.tanh(x), self.dropout, training=self.training)

        bsz = x.size(0)
        max_neighbor = torch.zeros(bsz, x.size(1))
        for i in range(int(self.label_num)):
            for j in range(bsz):
                for k in range(x.size(1)):
                    max_neighbor[j, k] += torch.sum(adjs[j, i, k, :])
        max_neighbor = int(torch.max(max_neighbor))

        supports = torch.zeros(bsz, x.size(1), self.label_num, max_neighbor, int(self.hidden_dim/2)).to(device)
        for i in range(int(self.label_num)):
            for j in range(bsz):
                for k in range(x.size(1)):
                    supports[j, k, i, :int(torch.sum(adjs[j, i, k])), :] = \
                        torch.matmul(x[j, adjs[j, i, k] == 1, :], self.W[i])
        # [B, L, r, max_neighbor, H]
        supports = supports.view(-1, max_neighbor, int(self.hidden_dim/2))
        supports, _ = self.aggregator(supports)
        supports = supports[:, -1, :].view(bsz, x.size(1), self.label_num, self.hidden_dim)

        output = x + torch.mean(supports, dim=2)
        return {
            'x': output + x,
            'feature': feature,
            'adj': inputs['adj'],
        }


class GSAGE(nn.Module):
    def __init__(self, max_number, hidden_dim, label_num, gsage_layer=2, dropout=0.1):
        super(GSAGE, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num
        self.gsage_layer = gsage_layer
        self.embedding = nn.Embedding(max_number, hidden_dim - 2, padding_idx=0)
        self.gsages = nn.ModuleList([GSAGELayer(hidden_dim, label_num, dropout=dropout) for _ in range(gsage_layer)])
        self.distmult = DistMult(hidden_dim)

    def forward(self, inputs):
        x, feature, adj = inputs['x'], inputs['feature'], inputs['adj']
        embed = self.embedding(x)
        embed = torch.cat([embed, feature], dim=2)

        gsage_result = {'x': embed, 'feature': feature, 'adj': adj}
        for gsage in self.gsages:
            gsage_result = gsage(gsage_result)
        embed = gsage_result['x']    # [bsz, L, H]

        return self.distmult(embed, embed)    # [bsz x L x L, 2]

