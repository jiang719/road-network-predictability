import copy
import math
import pickle
from functools import cmp_to_key
import numpy as np
import torch

from tester.vec_tester import is_valid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def distance(n1, n2):
    lat = 111 * 1e3 * abs(n1['lat'] - n2['lat'])
    lon = 111 * 1e3 * abs(n1['lon'] - n2['lon']) * np.cos(n1['lat'])
    return np.sqrt(lon**2 + lat**2)
'''


def angle(n1, n2):
    x1, y1 = n1['lon'], n1['lat']
    x2, y2 = n2['lon'], n2['lat']
    if x1 == x2 and y1 <= y2:
        return 0
    if x1 == x2 and y1 > y2:
        return 4
    k = (y2 - y1) / (x2 - x1)
    a = math.degrees(math.atan(k))
    if a >= 67.5:
        return 0 if y1 < y2 else 4
    elif a >= 22.5:
        return 1 if y1 < y2 else 5
    elif a >= -22.5:
        if a >= 0:
            return 2 if y1 < y2 else 6
        else:
            return 6 if y1 < y2 else 2
    elif a >= -67.5:
        return 7 if y1 < y2 else 3
    else:
        return 0 if y1 < y2 else 4


def edge_label(n1, n2):
    '''
    dist = distance(n1, n2)
    if dist <= 100:
        dist = 0
    elif dist <= 300:
        dist = 1
    else:
        dist = 2
    '''
    ang = angle(n1, n2)
    #return str(ang) + str(dist)
    return str(ang)


def get_edge_labels():
    labels = {}
    for ang in range(8):
        #for dist in range(3):
        #    k = str(ang) + str(dist)
        #    labels[k] = len(labels)
        k = str(ang)
        labels[k] = len(labels)
    return labels


def compare(n1, n2):
    if n1['lon'] == n2['lon']:
        return n1['lat'] - n2['lat']
    return n1['lon'] - n2['lon']


class GNNTester():
    def __init__(self, test_data, city):
        self.test_loader = test_data
        self.city = city
        self.id2node = {}
        self.initialize()

    def initialize(self):
        for k, v in self.test_loader.data[self.city].items():
            for node in v['nodes']:
                ids = node['osmid']
                if ids not in self.id2node:
                    self.id2node[ids] = node

    def prepare_batch_data(self, data, max_number, edge_labels):
        X = torch.zeros(len(data), max_number).long().to(device)
        F = torch.zeros(len(data), max_number, 2).to(device)
        A = torch.zeros(len(data), len(edge_labels), max_number, max_number).to(device)
        N, S, T = [], [], []
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
            N.append(nodes)
            S.append(source_edges)
            T.append(target_edges)

            id2index = {n['osmid']: i for i, n in enumerate(nodes)}

            x = [i + 1 for i in range(len(nodes))]
            x += [0] * (max_number - len(x))
            f = [[n['lon'], n['lat']] for n in nodes]
            f += [[0, 0] for i in range(max_number - len(f))]
            x = torch.LongTensor(x).to(device)
            f = torch.Tensor(f).to(device)
            adj = torch.zeros(len(edge_labels), max_number, max_number).to(device)

            for edge in source_edges:
                start, end = id2index[edge['start']], id2index[edge['end']]
                l1 = edge_labels[edge_label(nodes[start], nodes[end])]
                l2 = edge_labels[edge_label(nodes[end], nodes[start])]
                adj[l1, start, end] = 1.
                adj[l2, end, start] = 1.

            X[i], F[i], A[i] = x, f, adj
        return X, F, A, N, S, T

    def test(self, model, max_number, edge_labels, result_dir):
        model.eval()
        right, wrong, total = 0, 0, 0

        test_result = {}
        data = [self.test_loader[i] for i in range(len(self.test_loader))]
        batch_size = 32
        for _ in range(0, len(data), batch_size):
            X, F, A, N, S, T = self.prepare_batch_data(data[_: _ + batch_size], max_number, edge_labels)
            output = model({
                'x': X,
                'feature': F,
                'adj': A
            }).view(X.size(0), X.size(1), X.size(1), 2).to('cpu')
            for i in range(len(output)):
                ids = self.test_loader.ids[_ + i]
                predict = output[i][..., 1]
                existed_edges = S[i]
                cand_edges = []
                number = len(N[i])

                for j in range(number):
                    for k in range(j + 1, number):
                        start, end = N[i][j]['osmid'], N[i][k]['osmid']
                        if {'start': start, 'end': end} in T[i] or {'start': end, 'end': start} in T[i]:
                            target = 1
                        else:
                            target = 0
                        cand_edges.append({
                            'start': start,
                            'end': end,
                            'score': float(predict[j][k]),
                            'target': target,
                        })
                cand_edges.sort(key=lambda e: e['score'], reverse=True)
                test_result[ids] = cand_edges
                for edge in cand_edges:
                    if edge['score'] < np.log(0.5):
                        break
                    #if is_valid(edge, existed_edges, self.id2node):
                    existed_edges.append(edge)
                    if edge['target'] == 1:
                        right += 1
                    else:
                        wrong += 1
                total += len(T[i])
        precision = right / (right + wrong + 1e-9)
        recall = right / (total + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        pickle.dump(test_result, open(result_dir + self.city + '_result.pkl', 'wb'))
        return right, wrong, total, precision, recall, f1

