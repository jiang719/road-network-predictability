import copy
import random
from functools import cmp_to_key
import numpy as np
import torch
import torch.nn as nn

from tester.gnn_tester import get_edge_labels, compare, edge_label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNTrainer():
    def __init__(self, train_data, city, tester):
        self.train_loader = train_data
        self.tester = tester
        self.city = city
        self.max_number = self.get_max_number()
        self.edge_labels = get_edge_labels()
        self.model = None

    def get_max_number(self):
        max_number = 0
        for index in self.train_loader.data[self.city]:
            max_number = max(max_number, len(self.train_loader[index]['nodes']))
        for index in self.tester.test_loader.data[self.city]:
            max_number = max(max_number, len(self.tester.test_loader[index]['nodes']))
        return max_number + 1

    def prepare_batch_data(self, data):
        X = torch.zeros(len(data), self.max_number).long().to(device)
        F = torch.zeros(len(data), self.max_number, 2).to(device)
        A = torch.zeros(len(data), len(self.edge_labels), self.max_number, self.max_number).to(device)
        T = torch.zeros(len(data), self.max_number, self.max_number).long().to(device)
        for i, d in enumerate(data):
            nodes = copy.deepcopy(d['nodes'])
            min_lon = np.min([n['lon'] for n in nodes])
            max_lon = np.max([n['lon'] for n in nodes])
            min_lat = np.min([n['lat'] for n in nodes])
            max_lat = np.max([n['lat'] for n in nodes])
            for n in nodes:
                n['lon'] = (n['lon'] - min_lon) / (max_lon - min_lon)
                n['lat'] = (n['lat'] - min_lat) / (max_lat - min_lat)
            nodes.sort(key=cmp_to_key(compare))
            source_edges = copy.deepcopy(d['source_edges'])
            target_edges = copy.deepcopy(d['target_edges'])
            id2index = {n['osmid']: i for i, n in enumerate(nodes)}

            x = [i+1 for i in range(len(nodes))]
            x += [0] * (self.max_number - len(x))
            f = [[n['lon'], n['lat']] for n in nodes]
            f += [[0, 0] for i in range(self.max_number - len(f))]
            x = torch.LongTensor(x).to(device)
            f = torch.Tensor(f).to(device)
            adj = torch.zeros(len(self.edge_labels), self.max_number, self.max_number).to(device)
            target = torch.zeros(self.max_number, self.max_number).long().to(device)

            for edge in source_edges:
                start, end = id2index[edge['start']], id2index[edge['end']]
                l1 = self.edge_labels[edge_label(nodes[start], nodes[end])]
                l2 = self.edge_labels[edge_label(nodes[end], nodes[start])]
                adj[l1, start, end] = 1.
                adj[l2, end, start] = 1.
                target[start, end] = -1
                target[end, start] = -1

            for edge in target_edges:
                start, end = id2index[edge['start']], id2index[edge['end']]
                target[start, end] = 1
                target[end, start] = 1
            number = len(nodes)
            target[:, number:] = -1
            target[number:, :] = -1

            X[i], F[i], A[i], T[i] = x, f, adj, target
        return X, F, A, T

    def train_model(self, batch_size=4, epochs=7, result_dir=None):
        print('train data:', len(self.train_loader))
        if len(self.train_loader) < 100:
            batch_size = 2
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        loss_fct = nn.NLLLoss()
        data = [self.train_loader[i] for i in range(len(self.train_loader))]
        best_model = None
        best_f1 = 0.
        for epoch in range(epochs):
            self.model.train()
            if epoch == 4:
                for param in optimizer.param_groups:
                    param['lr'] = 0.02
            random.shuffle(data)
            epoch_loss = 0
            right, wrong, total = 0, 0, 0
            for i in range(0, len(data), batch_size):
                X, F, A, T = self.prepare_batch_data(data[i: i + batch_size])
                output = self.model({
                    'x': X,
                    'feature': F,
                    'adj': A
                })
                output = output.view(-1, 2)
                T = T.view(-1)

                index = (T == 0).nonzero().squeeze(-1)
                T0 = T.index_select(0, index)
                output0 = output.index_select(0, index)
                index = (T == 1).nonzero().squeeze(-1)
                T1 = T.index_select(0, index)
                output1 = output.index_select(0, index)
                T_ = torch.cat([T1, T0] + [T1 for _ in range(1, int(len(T0) / len(T1) / 4))], dim=0)
                output_ = torch.cat([output1, output0] +
                                    [output1 for _ in range(1, int(len(T0) / len(T1) / 4))], dim=0)

                optimizer.zero_grad()
                loss = loss_fct(output_, T_)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                index = (T == 1).nonzero().squeeze(-1)
                right += (output.index_select(0, index)[:, 1] >
                          float(torch.log(torch.Tensor([0.5]).to(device)))).nonzero().size(0)
                index = (T == 0).nonzero().squeeze(-1)
                wrong += (output.index_select(0, index)[:, 1] >
                          float(torch.log(torch.Tensor([0.5]).to(device)))).nonzero().size(0)
                total += (T == 1).nonzero().size(0)

            precision = right / (right + wrong + 1e-9)
            recall = right / (total + 1e-9)
            f1 = 2 * recall * precision / (recall + precision + 1e-9)
            print('epoch: {}, loss: {}, right: {}, wrong: {}, precision: {}, recall: {}, f1: {}'.format(
                epoch + 1, round(epoch_loss, 4), right, wrong, round(precision, 4), round(recall, 4), round(f1, 4)
            ))
            right, wrong, total, precision, recall, f1 = \
                self.tester.test(self.model, self.max_number, self.edge_labels, result_dir)
            print('test, right: {}, wrong: {}, total:{}, precision: {}, recall: {}, f1: {}'.format(
                right, wrong, total, round(precision, 4), round(recall, 4), round(f1, 4)
            ))
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(self.model)
        self.save_model(best_model)
        right, wrong, total, precision, recall, f1 = \
            self.tester.test(best_model, self.max_number, self.edge_labels, result_dir)
        print('final f1:', f1)

    def save_model(self, model):
        raise NotImplementedError

