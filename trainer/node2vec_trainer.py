import pickle
from abc import ABC

from models.node2vec import Node2Vec
from tester.vec_tester import VecTester
from trainer.vec_trainer import VecTrainer
from utils.data_loader import DataLoader


class Node2VecTrainer(VecTrainer):
    def __init__(self, embed_dim, train_data, city, tester):
        super().__init__(embed_dim, train_data, city, tester)
        self.vec_model = Node2Vec(num_walks=400)

    def save_distmult(self):
        obj = {
            'embed_dim': self.embed_dim,
            'city': self.city,
            'distmult': self.distmult,
        }
        pickle.dump(obj, open(data_dir + 'data_2020715/node2vec/models/' +
                              self.city + '_distmult.pkl', 'wb'))


if __name__ == "__main__":
    data_dir = 'E:/python-workspace/CityRoadPrediction/'
    train = DataLoader(data_dir + 'data_2020715/train/')
    test = DataLoader(data_dir + 'data_2020715/test/')

    cities = set(train.cities) & set(test.cities)
    cities = sorted(list(cities))
    for city in cities:
        print(city)
        train.initialize()
        train.load_dir_datas(city)
        test.initialize()
        test.load_dir_datas(city)
        tester = VecTester(embed_dim=50, test_data=test, city=city, data_dir=data_dir + 'data_2020715/node2vec/')
        trainer = Node2VecTrainer(embed_dim=50, train_data=train, city=city, tester=tester)
        trainer.prepare_train_embedding(data_dir + 'data_2020715/node2vec/')
        #trainer.train_distmult(data_dir=data_dir + 'data_2020715/node2vec/',
        #                       result_dir=data_dir + 'data_20200715/node2vec/result/')


