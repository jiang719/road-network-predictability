import pickle
import torch

from models.relational_gcn import RGCN
from tester.gnn_tester import GNNTester
from trainer.gnn_trainer import GNNTrainer
from utils.data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RGCNTrainer(GNNTrainer):
    def __init__(self, train_data, city, tester):
        super().__init__(train_data, city, tester)
        self.model = RGCN(
            max_number=self.max_number,
            hidden_dim=50,
            relation_num=len(self.edge_labels),
            v_num=4,
            gcn_layer=3,
            dropout=0.
        ).to(device)

    def save_model(self, best_model):
        obj = {
            'max_number': self.max_number,
            'hidden_dim': self.model.hidden_dim,
            'relation_num': self.model.relation_num,
            'v_num': self.model.v_num,
            'gcn_layer': self.model.gcn_layer,
            'city': self.city,
            'rgcn': best_model,
        }
        pickle.dump(obj, open(data_dir + 'data_20200610/relational-gcn/models/' +
                              self.city + '_distmult.pkl', 'wb'))


if __name__ == "__main__":
    data_dir = 'E:/python-workspace/CityRoadPrediction/'
    train = DataLoader(data_dir + 'data_20200610/train/')
    test = DataLoader(data_dir + 'data_20200610/test/')

    cities = set(train.cities) & set(test.cities)
    cities = sorted(list(cities))
    for city in cities:
        print(city)
        train.initialize()
        train.load_dir_datas(city)
        test.initialize()
        test.load_dir_datas(city)
        tester = GNNTester(test_data=test, city=city)
        trainer = RGCNTrainer(train, city, tester)
        trainer.train_model(result_dir='E:/python-workspace/CityRoadPrediction/data_20200610/relational-gcn/result/')


