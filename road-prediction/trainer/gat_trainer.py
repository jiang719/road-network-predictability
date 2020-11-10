import pickle

import torch

from models.gat import GAT
from tester.gnn_tester import GNNTester
from trainer.gnn_trainer import GNNTrainer
from utils.data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(6)


class GATTrainer(GNNTrainer):
    def __init__(self, train_data, city, tester):
        super().__init__(train_data, city, tester)
        self.model = GAT(
            max_number=self.max_number,
            hidden_dim=50,
            heads_num=8,
            label_num=len(self.edge_labels),
            gat_layer=2,
            dropout=0.1,
        ).to(device)

    def save_model(self, best_model):
        obj = {
            'max_number': self.max_number,
            'hidden_dim': self.model.hidden_dim,
            'heads_num': self.model.heads_num,
            'label_num': self.model.label_num,
            'gat_layer': self.model.gat_layer,
            'city': self.city,
            'gat': best_model,
        }
        pickle.dump(obj, open(data_dir + 'data_2020715/gat/models/' +
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
        tester = GNNTester(test_data=test, city=city)
        trainer = GATTrainer(train, city, tester)
        trainer.train_model(result_dir='E:/python-workspace/CityRoadPrediction/data_2020715/gat/result/')
