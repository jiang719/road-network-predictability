import pickle
import math
import torch
import copy
import numpy as np
from utils.data_loader import DataLoader


def is_intersect(node1, node2, node3, node4):
    lon1, lat1 = node1['lon'], node1['lat']
    lon2, lat2 = node2['lon'], node2['lat']
    lon3, lat3 = node3['lon'], node3['lat']
    lon4, lat4 = node4['lon'], node4['lat']
    distance1_3 = abs(lon1 - lon3) * 100000 + abs(lat1 - lat3) * 100000
    distance1_4 = abs(lon1 - lon4) * 100000 + abs(lat1 - lat4) * 100000
    distance2_3 = abs(lon2 - lon3) * 100000 + abs(lat2 - lat3) * 100000
    distance2_4 = abs(lon2 - lon4) * 100000 + abs(lat2 - lat4) * 100000
    min_distance = np.min([distance1_3, distance1_4, distance2_3, distance2_4])
    if min_distance == 0:
        return False
    else:
        if np.max([lon1, lon2]) < np.min([lon3, lon4]) or np.max([lon3, lon4]) < np.min([lon1, lon2]):
            return False
        else:
            sort_points = np.sort([lon1, lon2, lon3, lon4])
            left_point, right_point = sort_points[1], sort_points[2]
            if lon1 == lon2:
                value_point1 = [lat1, lat2]
            else:
                value_point1 = [(lat2-lat1)/(lon2-lon1)*(left_point-lon1)+lat1, (lat2-lat1)/(lon2-lon1)*(right_point-lon1)+lat1]
            if lon3 == lon4:
                value_point2 = [lat3, lat4]
            else:
                value_point2 = [(lat4 - lat3) / (lon4 - lon3) * (left_point - lon3) + lat3,
                               (lat4 - lat3) / (lon4 - lon3) * (right_point - lon3) + lat3]
            if np.max(value_point1) < np.min(value_point2) or np.max(value_point2) < np.min(value_point1):
                return False
            else:
                return True


def is_acute(node1, node2, node3, node4):
    lon1, lat1 = node1['lon'], node1['lat']
    lon2, lat2 = node2['lon'], node2['lat']
    lon3, lat3 = node3['lon'], node3['lat']
    lon4, lat4 = node4['lon'], node4['lat']
    distance1_3 = abs(lon1-lon3)*100000 + abs(lat1-lat3)*100000
    distance1_4 = abs(lon1-lon4)*100000 + abs(lat1-lat4)*100000
    distance2_3 = abs(lon2-lon3)*100000 + abs(lat2-lat3)*100000
    distance2_4 = abs(lon2-lon4)*100000 + abs(lat2-lat4)*100000
    min_distance = np.min([distance1_3, distance1_4, distance2_3, distance2_4])
    if min_distance > 0:
        return False
    else:
        if distance1_3 == min_distance:
            x1,y1 = lon2-lon1, lat2-lat1
            x2,y2 = lon4-lon3, lat4-lat3
        if distance1_4 == min_distance:
            x1,y1 = lon2-lon1, lat2-lat1
            x2,y2 = lon3-lon4, lat3-lat4
        if distance2_3 == min_distance:
            x1,y1 = lon1-lon2, lat1-lat2
            x2,y2 = lon4-lon3, lat4-lat3
        if distance2_4 == min_distance:
            x1,y1 = lon1-lon2, lat1-lat2
            x2,y2 = lon3-lon4, lat3-lat4

        vector_1 = [x1, y1]
        vector_2 = [x2, y2]
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) / math.pi * 180
        if angle < 35:
            return True
        else:
            return False


def is_valid(new_edge, existed_edges, id2node):
    for edge in existed_edges:
        if is_intersect(
                id2node[new_edge['start']], id2node[new_edge['end']],
                id2node[edge['start']], id2node[edge['end']]
        ) or is_acute(
            id2node[new_edge['start']], id2node[new_edge['end']],
            id2node[edge['start']], id2node[edge['end']]
        ):
            return False
    return True


class VecTester():
    def __init__(self, embed_dim, test_data, city, data_dir):
        self.embed_dim = embed_dim
        self.test_loader = test_data
        self.city = city
        self.embedding = {}
        self.id2node = {}
        self.initialize(data_dir)

    def initialize(self, data_dir):
        for k, v in self.test_loader.data[self.city].items():
            for node in v['nodes']:
                ids = node['osmid']
                if ids not in self.id2node:
                    self.id2node[ids] = node
        self.embedding = pickle.load(open(data_dir + 'test/' + self.city + '_embedding.pkl', 'rb'))

    def test(self, model, result_dir):
        model.eval()
        model = model.to('cpu')
        right, wrong, total = 0, 0, 0

        test_result = {}
        for ids in self.embedding:
            existed_edges = copy.deepcopy(self.test_loader[ids]['source_edges'])
            cand_edges = []
            batch_size = 101
            for i in range(0, len(self.embedding[ids]), batch_size):
                start = torch.Tensor([sample['start_embedding'].tolist()
                                      for sample in self.embedding[ids][i: i + batch_size]])
                end = torch.Tensor([sample['end_embedding'].tolist()
                                    for sample in self.embedding[ids][i: i + batch_size]])

                output = model(start, end).squeeze(0)
                for j in range(output.size(0)):
                    sample = self.embedding[ids][i + j]
                    edge = {'start': int(sample['start_id']), 'end': int(sample['end_id']),
                            'score': float(output[j][1]), 'target': sample['target']}
                    cand_edges.append(edge)
            cand_edges.sort(key=lambda e: e['score'], reverse=True)
            test_result[ids] = cand_edges
            for edge in cand_edges:
                if edge['score'] < np.log(0.5):
                    break
                #if is_valid(edge, existed_edges, self.id2node):
                existed_edges.append(edge)
                if {'start': edge['start'], 'end': edge['end']} in self.test_loader[ids]['target_edges'] or \
                        {'start': edge['end'], 'end': edge['start']} in self.test_loader[ids]['target_edges']:
                    right += 1
                else:
                    wrong += 1

            total += len(self.test_loader[ids]['target_edges'])
        precision = right / (right + wrong + 1e-9)
        recall = right / (total + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        pickle.dump(test_result, open(result_dir + self.city + '_result.pkl', 'wb'))
        return right, wrong, total, precision, recall, f1


if __name__ == "__main__":
    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/test/')
    test.load_dir_datas('Akron')
    tester = VecTester(embed_dim=50, test_data=test, city='Akron')
    print(test[0]['source_edges'])
