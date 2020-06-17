import pickle
import random
import torch
import torch.nn as nn

from models.distmult import DistMult
from models.node2vec import Node2Vec
from tester.node2vec_tester import Node2VecTester
from utils.data_loader import DataLoader


class Node2VecTrainer():
    def __init__(self, embed_dim, train_data, city, tester):
        self.embed_dim = embed_dim
        self.train_loader = train_data
        self.tester = tester
        self.city = city
        self.node2vec = Node2Vec(num_walks=20 * len(self.train_loader.data[self.city]))
        self.distmult = DistMult(embed_dim)
        self.embedding = []

    def prepare_train_embedding(self):
        nodes, edges = [], []
        for index in self.train_loader.data[self.city]:
            nodes += self.train_loader[index]['nodes']
            edges += self.train_loader[index]['source_edges']
        for index in self.tester.test_loader.data[self.city]:
            nodes += self.tester.test_loader[index]['nodes']
            edges += self.tester.test_loader[index]['source_edges']
        G = DataLoader.build_graph(nodes, edges)
        self.node2vec.build_model(G)
        embeds = self.node2vec.train(embed_size=self.embed_dim)
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
                    'start_embedding': embeds[str(start)],
                    'end_embedding': embeds[str(end)],
                    'target': target,
                })
        pickle.dump(self.embedding,
                    open('E:/python-workspace/CityRoadPrediction/data_20200610/node2vec/train/' +
                         self.city + '_embedding.pkl', 'wb'))

        test_embedding = []
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
            for (start, end, target) in samples:
                test_embedding.append({
                    'start_id': str(start),
                    'end_id': str(end),
                    'start_embedding': embeds[str(start)],
                    'end_embedding': embeds[str(end)],
                    'target': target,
                })
        pickle.dump(test_embedding,
                    open('E:/python-workspace/CityRoadPrediction/data_20200610/node2vec/test/' +
                         self.city + '_embedding.pkl', 'wb'))

    def train_distmult(self, batch_size=32, epochs=10):
        samples = pickle.load(open('E:/python-workspace/CityRoadPrediction/data_20200610/node2vec/train/' +
                                   self.city + '_embedding.pkl', 'rb'))
        positive = [s for s in samples if s['target'] == 1]
        negative = [s for s in samples if s['target'] == 0]
        self.embedding = positive + negative
        for _ in range(1, int(len(negative) / len(positive))):
            self.embedding += positive

        optimizer = torch.optim.SGD(self.distmult.parameters(), lr=0.01, momentum=0.9)
        loss_fct = nn.NLLLoss()
        #print('train data:', len(self.embedding))
        for epoch in range(epochs):
            self.distmult.train()
            self.embedding = random.sample(negative, len(positive)) + positive
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

            right, wrong, total, precision, recall, f1 = self.tester.test(self.distmult)
            print('test, right: {}, wrong: {}, total:{}, precision: {}, recall: {}, f1: {}'.format(
                right, wrong, total, round(precision, 4), round(recall, 4), round(f1, 4)
            ))

    def save_distmult(self):
        obj = {
            'embed_dim': self.embed_dim,
            'city': self.city,
            'distmult': self.distmult,
        }
        pickle.dump(obj, open('E:/python-workspace/CityRoadPrediction/data_20200610/node2vec/' +
                              self.city + '_distmult.pkl', 'wb'))


if __name__ == "__main__":
    data_dir = 'E:/python-workspace/CityRoadPrediction/'
    train = DataLoader(data_dir + 'data_20200610/train/')
    test = DataLoader(data_dir + 'data_20200610/test/')
    for city in train.data:
        if city not in test.data:
            continue
        print(city)
        train.load_dir_datas(city)
        test.load_dir_datas(city)
        tester = Node2VecTester(embed_dim=100, test_data=test, city=city)
        trainer = Node2VecTrainer(embed_dim=100, train_data=train, city=city, tester=tester)
        trainer.prepare_train_embedding()
        #trainer.train_distmult()
