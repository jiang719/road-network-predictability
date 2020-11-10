import torch
import torch.nn as nn
import torch.nn.functional as F

from models.distmult import DistMult

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RGCNLayer(nn.Module):
    def __init__(self, hidden_dim, relation_num, v_num, dropout=0.1):
        super(RGCNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_num = relation_num
        self.fc = nn.Linear(hidden_dim + 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W = nn.Parameter(torch.rand(hidden_dim, v_num, hidden_dim).to(device))
        self.a = nn.Parameter(torch.rand(relation_num + 1, v_num).to(device))
        self.dropout = dropout

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)

    def forward(self, inputs):
        x, feature, _adjs = inputs['x'], inputs['feature'], inputs['adj']

        x = torch.cat([x, feature], dim=2)
        #print(x.size())
        x = self.fc(x)
        x = self.norm(x)
        x = F.dropout(torch.tanh(x), self.dropout)

        bsz = x.size(0)
        supports = []
        adjs = [torch.eye(x.size(1)).repeat(bsz, 1, 1).to(device)]
        for i in range(int(self.relation_num)):
            adjs.append(_adjs[:, i, :, :])

        for adj in adjs:
            supports.append(torch.bmm(adj, x).unsqueeze(1))
            # [bsz, L, L] x [bsz, L, H] = [bsz, L, H] -> [bsz, 1, L, H]
        supports = torch.cat(supports, dim=1)  # [bzs, 1+relation_num, L, H]

        output = torch.matmul(self.a, self.W).permute(1, 0, 2)  # [1+r, v] x [H, v, H] = [H, 1+r, H] -> [1+r, H, H]
        output = torch.matmul(supports, output)     # [bsz, 1+r, L, H] x [1+r, H, H] = [bsz, 1+r, L, H]
        output = torch.sum(output, dim=1)
        #output /= (self.relation_num + 1)       # [bsz, L, H]
        output = F.dropout(output, self.dropout)

        return {
            'x': output + x,
            'feature': feature,
            'adj': inputs['adj'],
        }


class RGCN(nn.Module):
    def __init__(self, max_number, hidden_dim, relation_num, v_num, gcn_layer=2, dropout=0.1):
        super(RGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_num = relation_num
        self.v_num = v_num
        self.gcn_layer = gcn_layer
        self.embedding = nn.Embedding(max_number, hidden_dim-2, padding_idx=0)
        self.gcns = nn.ModuleList([RGCNLayer(hidden_dim, relation_num, v_num, dropout)
                                   for _ in range(gcn_layer)])
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

