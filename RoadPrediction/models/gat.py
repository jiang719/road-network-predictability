import torch
import torch.nn as nn
import torch.nn.functional as F
from models.distmult import DistMult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_num, dropout=0.1):
        super(GATLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(input_dim + 2, hidden_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.norm = nn.LayerNorm(hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(label_num + 1, 2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        #self.r = nn.Parameter(torch.rand(label_num + 1))

    def forward(self, inputs):
        x, feature, adjs = inputs['x'], inputs['feature'], inputs['adj']

        x = torch.cat([x, feature], dim=2)
        x = torch.matmul(x, self.W)     # [B, L, H]
        x = self.norm(x)
        x = F.dropout(torch.tanh(x), p=self.dropout, training=self.training)

        bsz, num = x.size(0), x.size(1)
        h = torch.cat([x.repeat(1, 1, num).view(bsz, num*num, -1),
                       x.repeat(1, num, 1)], dim=2).view(bsz*num*num, 1, -1)     # [BxLxL, 1, 2H]
        #h = torch.matmul(h, self.a).squeeze(-1)                  # [BxLxL, 1]
        h = torch.cat([torch.matmul(h, self.a[i]).squeeze(-1)
                       for i in range(self.label_num + 1)], dim=1)  # [BxLxL, label_num+1]
        h = h.view(bsz, num, num, -1).permute(0, 3, 1, 2)       # [B, label_num+1, L, L]
        e = torch.tanh(h)

        attention = -9e15 * torch.ones(bsz, num, num).to(device)
        attention = torch.where(
            torch.eye(num).repeat(bsz, 1, 1).to(device) > 0, e[:, -1, :, :], attention
        )
        for i in range(self.label_num):
            attention = torch.where(adjs[:, i, :, :] > 0, e[:, i, :, :], attention)
        attention = F.softmax(attention, dim=2)
        output = torch.bmm(attention, x)

        return output


class MultiGATLayer(nn.Module):
    def __init__(self, hidden_dim, heads_num, label_num, dropout=0.1):
        super(MultiGATLayer, self).__init__()
        self.inner_dim = int(hidden_dim / 2)
        self.attentions = [GATLayer(hidden_dim, self.inner_dim, label_num, dropout)
                           for _ in range(heads_num)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.dropout = dropout
        self.fc = nn.Linear(self.inner_dim * heads_num, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        x = torch.cat([att(inputs) for att in self.attentions], dim=2)
        x = torch.tanh(self.norm(self.fc(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return {
            'x': x + inputs['x'],
            'feature': inputs['feature'],
            'adj': inputs['adj']
        }


class GAT(nn.Module):
    def __init__(self, max_number, hidden_dim, heads_num, label_num, gat_layer=3, dropout=0.1):
        super(GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        self.label_num = label_num
        self.gat_layer = gat_layer
        self.embedding = nn.Embedding(max_number, hidden_dim-2, padding_idx=0)
        self.gats = nn.ModuleList([MultiGATLayer(hidden_dim, heads_num, label_num, dropout)
                                   for _ in range(self.gat_layer)])
        self.distmult = DistMult(hidden_dim)

    def forward(self, inputs):
        x, feature, adj = inputs['x'], inputs['feature'], inputs['adj']
        embed = self.embedding(x)
        embed = torch.cat([embed, feature], dim=2)

        gat_result = {'x': embed, 'feature': feature, 'adj': adj}
        for gat in self.gats:
            gat_result = gat(gat_result)
        embed = gat_result['x']

        return self.distmult(embed, embed)  # [bsz x L x L, 2]

