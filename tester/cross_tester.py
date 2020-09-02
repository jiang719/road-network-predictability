import json
import pickle

from tester.gnn_tester import GNNTester, get_edge_labels
from utils.data_loader import DataLoader


def cross_f1(train, test):
    cities = set(train.cities) & set(test.cities)
    cities = sorted(list(cities))

    result = {}
    for c1 in cities:
        print('model:', c1)
        result[c1] = {}
        load = pickle.load(open(data_dir + 'data_2020715/relational-gcn/models/' + c1 + '_distmult.pkl', 'rb'))
        model = load['rgcn']
        for c2 in cities:
            test.initialize()
            test.load_dir_datas(c2)
            tester = GNNTester(test_data=test, city=c2)
            max_number = tester.get_max_number()

            f1 = tester.improved_test(model, max_number, get_edge_labels(), int(load['max_number']))
            result[c1][c2] = f1
            print(c1, c2, f1)
    json.dump(result, open(data_dir + 'cross_f1score.json', 'w'), indent=2)


def corss_sample_f1(train, test):
    cities = set(train.cities) & set(test.cities)
    cities = sorted(list(cities))

    result = {}
    for c1 in cities:
        result[c1] = {}
        load = pickle.load(open(data_dir + 'data_2020715/relational-gcn/models/' + c1 + '_distmult.pkl', 'rb'))
        model = load['rgcn']
        for c2 in cities:
            if c2 == c1:
                continue
            result[c1][c2] = {}

            train.initialize()
            train.load_dir_datas(c2)
            tester = GNNTester(test_data=train, city=c2)
            max_number = tester.get_max_number()
            f1, res = tester.improved_test(model, max_number, get_edge_labels(), int(load['max_number'])-1)
            result[c1][c2].update(res)

            test.initialize()
            test.load_dir_datas(c2)
            tester = GNNTester(test_data=test, city=c2)
            max_number = tester.get_max_number()
            f1, res = tester.improved_test(model, max_number, get_edge_labels(), int(load['max_number'])-1)
            result[c1][c2].update(res)
        json.dump(result, open(data_dir + 'cross_samples_f1score.json', 'w'), indent=2)
    json.dump(result, open(data_dir + 'cross_samples_f1score.json', 'w'), indent=2)


if __name__ == "__main__":
    data_dir = 'D:/data/road-network-predictability/'
    train = DataLoader(data_dir + 'data_2020715/train/')
    test = DataLoader(data_dir + 'data_2020715/test/')
    corss_sample_f1(train, test)

