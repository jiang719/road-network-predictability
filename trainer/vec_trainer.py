import copy
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from models.distmult import DistMult
from utils.data_loader import DataLoader


class VecTrainer():
    def __init__(self, embed_dim, train_data, city, tester):
        self.embed_dim = embed_dim
        self.train_loader = train_data
        self.tester = tester
        self.city = city
        self.distmult = DistMult(embed_dim)
        self.embedding = []
        self.vec_model = None

    def prepare_train_embedding(self, data_dir):
        data = copy.deepcopy(self.train_loader.data[self.city])
        data.update(self.tester.test_loader.data[self.city])
        keys = sorted(list(data.keys()))
        embeds = {}
        for i in range(0, len(keys), 30):
            print(self.city, i, len(keys))
            nodes, edges = [], []
            for index in keys[i: i + 30]:
                nodes += data[index]['nodes']
                edges += data[index]['source_edges']
            G = DataLoader.build_graph(nodes, edges)
            self.vec_model.build_model(G)
            embeds.update(self.vec_model.train(embed_size=self.embed_dim))
        for index in self.train_loader.data[self.city]:
            positive, negative = [], []
            sample = self.train_loader.data[self.city][index]
            for i, n1 in enumerate(sample['nodes']):
                for j, n2 in enumerate(sample['nodes'][i + 1:]):
                    if {'start': n1['osmid'], 'end': n2['osmid']} in sample['target_edges'] or \
                            {'start': n2['osmid'], 'end': n1['osmid']} in sample['target_edges']:
                        positive.append([n1['osmid'], n2['osmid'], 1])
                    elif {'start': n1['osmid'], 'end': n2['osmid']} not in sample['source_edges'] and \
                            {'start': n2['osmid'], 'end': n1['osmid']} not in sample['source_edges']:
                        negative.append([n1['osmid'], n2['osmid'], 0])
            samples = positive + negative
            for (start, end, target) in samples:
                self.embedding.append({
                    'start_id': str(start),
                    'end_id': str(end),
                    'start_embedding': embeds[str(start)] if str(start) in embeds else np.zeros(self.embed_dim),
                    'end_embedding': embeds[str(end)] if str(end) in embeds else np.zeros(self.embed_dim),
                    'target': target,
                })
        pickle.dump(self.embedding,
                    open(data_dir + 'train/' + self.city + '_embedding.pkl', 'wb'))

        test_embedding = {}
        for index in self.tester.test_loader.data[self.city]:
            positive, negative = [], []
            sample = self.tester.test_loader.data[self.city][index]
            for i, n1 in enumerate(sample['nodes']):
                for j, n2 in enumerate(sample['nodes'][i + 1:]):
                    if {'start': n1['osmid'], 'end': n2['osmid']} in sample['target_edges'] or \
                            {'start': n2['osmid'], 'end': n1['osmid']} in sample['target_edges']:
                        positive.append([n1['osmid'], n2['osmid'], 1])
                    elif {'start': n1['osmid'], 'end': n2['osmid']} not in sample['source_edges'] and \
                            {'start': n2['osmid'], 'end': n1['osmid']} not in sample['source_edges']:
                        negative.append([n1['osmid'], n2['osmid'], 0])
            samples = positive + negative
            test_embedding[index] = []
            for (start, end, target) in samples:
                test_embedding[index].append({
                    'start_id': str(start),
                    'end_id': str(end),
                    'start_embedding': embeds[str(start)] if str(start) in embeds else np.zeros(self.embed_dim),
                    'end_embedding': embeds[str(end)] if str(end) in embeds else np.zeros(self.embed_dim),
                    'target': target,
                })
        pickle.dump(test_embedding,
                    open(data_dir + 'test/' + self.city + '_embedding.pkl', 'wb'))

    def train_distmult(self, batch_size=128, epochs=7, data_dir=None, result_dir=None):
        samples = pickle.load(open(data_dir + 'train/' + self.city + '_embedding.pkl', 'rb'))
        #test_data = []
        #for k, v in self.tester.embedding.items():
        #    test_data += [str(s['start_id']) + '_' + str(s['end_id']) for s in v] + \
        #                 [str(s['end_id']) + '_' + str(s['start_id']) for s in v]
        #test_data = set(test_data)

        #positive = [s for s in samples if s['target'] == 1 and
        #            str(s['start_id']) + '_' + str(s['end_id']) not in test_data]
        #negative = [s for s in samples if s['target'] == 0 and
        #            str(s['start_id']) + '_' + str(s['end_id']) not in test_data]
        positive = [s for s in samples if s['target'] == 1]
        negative = [s for s in samples if s['target'] == 0]
        self.embedding = positive + negative
        for _ in range(1, int(len(negative) / len(positive) / 4)):
            self.embedding += positive
        print('train data:', len(self.embedding))

        optimizer = torch.optim.SGD(self.distmult.parameters(), lr=0.01, momentum=0.9)
        loss_fct = nn.NLLLoss()
        best_model = None
        best_f1 = -1.
        for epoch in range(epochs):
            self.distmult.train()
            random.shuffle(self.embedding)
            epoch_loss = 0
            right, wrong, total = 0, 0, 0
            for i in range(0, len(self.embedding), batch_size):
                starts = [torch.Tensor(e['start_embedding'].tolist()).unsqueeze(0)
                          for e in self.embedding[i: i + batch_size]]
                ends = [torch.Tensor(e['end_embedding'].tolist()).unsqueeze(0)
                        for e in self.embedding[i: i + batch_size]]
                targets = torch.LongTensor([e['target'] for e in self.embedding[i: i + batch_size]])
                starts = torch.cat(starts, dim=0)
                ends = torch.cat(ends, dim=0)

                output = self.distmult(starts, ends)

                optimizer.zero_grad()
                loss = loss_fct(output, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                output = output.to('cpu')
                index = (targets == 1).nonzero().squeeze(-1)
                right += (output.index_select(0, index)[:, 1] >
                          float(torch.log(torch.Tensor([0.5])))).nonzero().size(0)
                index = (targets == 0).nonzero().squeeze(-1)
                wrong += (output.index_select(0, index)[:, 1] >
                          float(torch.log(torch.Tensor([0.5])))).nonzero().size(0)
                total += (targets == 1).nonzero().size(0)

            precision = right / (right + wrong + 1e-9)
            recall = right / (total + 1e-9)
            f1 = 2 * recall * precision / (recall + precision + 1e-9)
            print('epoch: {}, loss: {}, right: {}, wrong: {}, precision: {}, recall: {}, f1: {}'.format(
                epoch + 1, round(epoch_loss, 4), right, wrong, round(precision, 4), round(recall, 4), round(f1, 4)
            ))

            right, wrong, total, precision, recall, f1 = self.tester.test(self.distmult, result_dir)
            print('test, right: {}, wrong: {}, total:{}, precision: {}, recall: {}, f1: {}'.format(
                right, wrong, total, round(precision, 4), round(recall, 4), round(f1, 4)
            ))
            if f1 > best_f1 and epoch >= 3:
                best_f1 = f1
                best_model = copy.deepcopy(self.distmult)
        self.save_model(best_model)
        right, wrong, total, precision, recall, f1 = self.tester.test(best_model, result_dir)
        print('final f1:', f1)

    def save_model(self, model):
        raise NotImplementedError

