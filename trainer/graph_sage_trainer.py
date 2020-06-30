import torch
import pickle

from models.graph_sage import GSAGE
from tester.gnn_tester import GNNTester
from trainer.gnn_trainer import GNNTrainer
from utils.data_loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GSAGETrainer(GNNTrainer):
    def __init__(self, train_data, city, tester):
        super().__init__(train_data, city, tester)
        self.model = GSAGE(
            max_number=self.max_number,
            hidden_dim=50,
            label_num=len(self.edge_labels),
            gsage_layer=2,
            dropout=0.1,
        ).to(device)

    def save_model(self, best_model):
        obj = {
            'max_number': self.max_number,
            'hidden_dim': self.model.hidden_dim,
            'label_num': self.model.label_num,
            'gsage_layer': self.model.gsage_layer,
            'city': self.city,
            'gsage': best_model,
        }
        pickle.dump(obj, open(data_dir + 'data_20200610/graph-sage/models/' +
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
        trainer = GSAGETrainer(train, city, tester)
        trainer.train_model(result_dir='E:/python-workspace/CityRoadPrediction/data_20200610/graph-sage/result/')
