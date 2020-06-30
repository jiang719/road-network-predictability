import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.distmult import DistMult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_adj(adj):
    adj_ = copy.deepcopy(adj)
    #bsz, max_number = adj_.size(0), adj_.size(1)
    #adj_ += torch.eye(max_number).repeat(bsz, 1, 1).to(device)
    rowsum = adj_.sum(-1)
    degree_mat_inv_sqrt = torch.diag_embed(torch.pow(rowsum, -0.5))
    return torch.bmm(torch.bmm(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)


class SGCNLayer(nn.Module):
    def __init__(self, hidden_dim, label_num, v_num=4, dropout=0.1):
        super(SGCNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num
        self.dropout = dropout

        self.fc = nn.Linear(hidden_dim + 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W = nn.Parameter(torch.rand(hidden_dim, v_num, hidden_dim).to(device))
        self.a = nn.Parameter(torch.rand(label_num + 1, v_num).to(device))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, inputs):
        x, feature, adjs = inputs['x'], inputs['feature'], inputs['adj']

        x = torch.cat([x, feature], dim=2)
        x = self.fc(x)  # [B, L, H]
        x = self.norm(x)
        x = F.dropout(torch.tanh(x), p=self.dropout, training=self.training)

        bsz, max_number = adjs.size(0), adjs.size(2)
        supports = [torch.bmm(torch.eye(max_number).repeat(bsz, 1, 1), x).unsqueeze(1)]
        for i in range(self.label_num):
            supports.append(torch.bmm(adjs[:, i, :, :], x).unsqueeze(1))
        supports = torch.cat(supports, dim=1)
        output = torch.matmul(self.a, self.W).permute(1, 0, 2)  # [1+r, v] x [H, v, H] = [H, 1+r, H] -> [1+r, H, H]
        output = torch.matmul(supports, output)  # [bsz, 1+r, L, H] x [1+r, H, H] = [bsz, 1+r, L, H]
        output = torch.sum(output, dim=1)
        '''
        output = torch.bmm(torch.eye(max_number).repeat(bsz, 1, 1), x) * self.a[-1]
        for i in range(self.label_num):
            output += torch.bmm(adjs[:, i, :, :], x) * self.a[i]
        '''
        output /= (self.label_num + 1)
        output = F.dropout(output, p=self.dropout, training=self.training)

        return {
            'x': output + inputs['x'],
            'feature': inputs['feature'],
            'adj': inputs['adj']
        }


class SGCN(nn.Module):
    def __init__(self, max_number, hidden_dim, label_num, gcn_layer=3, dropout=0.1):
        super(SGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_num = label_num
        self.gcn_layer = gcn_layer
        self.embedding = nn.Embedding(max_number, hidden_dim-2, padding_idx=0)
        self.gcns = nn.ModuleList([SGCNLayer(hidden_dim, label_num, dropout=dropout) for _ in range(gcn_layer)])
        self.distmult = DistMult(hidden_dim)

    def forward(self, inputs):
        x, feature, adj = inputs['x'], inputs['feature'], inputs['adj']
        embed = self.embedding(x)
        embed = torch.cat([embed, feature], dim=2)

        gcn_result = {'x': embed, 'feature': feature, 'adj': adj}
        for gcn in self.gcns:
            gcn_result = gcn(gcn_result)
        embed = gcn_result['x']    # [bsz, L, H]

        return self.distmult(embed, embed)    # [bsz x L x L, 2]

