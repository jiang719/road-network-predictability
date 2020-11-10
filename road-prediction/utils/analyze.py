import copy
import json
import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from tester.vec_tester import is_valid
from utils.data_loader import DataLoader


def load_city_result(city, model, data_dir):
    return pickle.load(open(data_dir + model + '/result/' + city + '_result.pkl', 'rb'))


def load_model_result(model, data_dir):
    files = os.listdir(data_dir + model + '/result/')
    result = {}
    for file in files:
        city = file.split('_')[0].strip()
        #if city not in ['Guangzhou']:
        #    continue
        result[city] = load_city_result(city, model, data_dir)
    return result


def roc(models):
    for i, model in enumerate(models):
        print(model)
        result = load_model_result(model.lower(), data_dir)
        y = []
        for city in result:
            for index, v in result[city].items():
                for sample in v:
                    y.append({
                        'score': sample['score'],
                        'target': int(sample['target'])
                    })
        del result
        y = sorted(y, key=lambda e: e['score'], reverse=True)
        y_score, y_label = [_['score'] for _ in y], [_['target'] for _ in y]
        print(len(y_score))
        fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score, pos_label=1)
        roc = metrics.roc_auc_score(y_label, y_score)

        plt.plot(fpr, tpr, label=model + ': ' + str(round(roc, 3)))
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve on Top10 cities')
    plt.show()


def precision_recall(models):
    def metrics(Y):
        positive = sum([y['target'] for y in Y])

        thresholds = np.linspace(1, 1e-9, 1000)
        precision, recall = [], []
        index = 0
        right, wrong = 0, 0
        #best_f1, best_threshold = 0., 0.
        for _, th in enumerate(thresholds):
            for i in range(index, len(Y)):
                if Y[i]['score'] < math.log(th):
                    index = i
                    break
                if Y[i]['target'] == 1:
                    right += 1
                else:
                    wrong += 1
            p = 1.0 * right / (right + wrong + 1e-9)
            r = 1.0 * right / positive
            precision.append(p)
            recall.append(r)
            #f1 = 2 * p * r / (p + r + 1e-9)
            #if f1 > best_f1:
            #    best_f1 = f1
            #    best_threshold = th

        pr_sort = {r: p for p, r in zip(precision, recall)}
        pr_sort.pop(0)
        pr_sort = [[p, r] for r, p in pr_sort.items()]
        pr_sort.sort(key=lambda e: e[1])
        precision, recall = [r[0] for r in pr_sort], [r[1] for r in pr_sort]
        return precision, recall

    for i, model in enumerate(models):
        print(model)
        result = load_model_result(model.lower(), data_dir)
        y = []
        for city in result:
            for index, v in result[city].items():
                for sample in v:
                    y.append({
                        'score': sample['score'],
                        'target': int(sample['target'])
                    })
        del result
        y = sorted(y, key=lambda e: e['score'], reverse=True)
        precision, recall = metrics(y)
        print(len(y))
        plt.plot(recall, precision, label=model)
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve on Top10 cities')
    plt.show()


def best_threshold(model):
    def metrics(Y):
        positive = sum([y['target'] for y in Y])

        thresholds = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
        index = 0
        right, wrong = 0, 0
        existed_edges = {ids: test[ids]['source_edges'] for ids in test.ids}
        id2node = {node['osmid']: node for ids in test.ids for node in test[ids]['nodes']}
        best_f1, best_th = 0, 0
        for _, th in enumerate(thresholds):
            for i in range(index, len(Y)):
                if Y[i]['score'] < math.log(th):
                    index = i
                    break
                if is_valid({'start': Y[i]['start'], 'end': Y[i]['end']}, existed_edges[Y[i]['id']], id2node):
                    existed_edges[Y[i]['id']].append({'start': Y[i]['start'], 'end': Y[i]['end']})
                    if Y[i]['target'] == 1:
                        right += 1
                    else:
                        wrong += 1
            p = 1.0 * right / (right + wrong + 1e-9)
            r = 1.0 * right / positive
            f1 = 2 * p * r / (p + r + 1e-9)
            if best_f1 < f1:
                best_f1 = f1
                best_th = th
                print(p, r, best_f1, best_th)
        return best_f1, best_th

    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/test/')
    test.load_all_datas()
    result = load_model_result(model.lower(), data_dir)
    y = []
    for city in result:
        for index, v in result[city].items():
            for sample in v:
                y.append({
                    'id': index,
                    'start': sample['start'],
                    'end': sample['end'],
                    'score': sample['score'],
                    'target': int(sample['target'])
                })
    del result
    y = sorted(y, key=lambda e: e['score'], reverse=True)
    f1, th = metrics(y)
    print(f1, th)


def predict(model):
    def metrics(Y, ids):
        positive = sum([y['target'] for y in Y])

        if city in ['Hongkong', 'Guangzhou', 'Singapore']:
            thresholds = 0.2
        elif city in ['Beijing', 'Shanghai', 'Shenzhen']:
            thresholds = 0.45
        else:
            thresholds = 0.6

        right, wrong = 0, 0
        existed_edges = test[ids]['source_edges']
        id2node = {node['osmid']: node for node in test[ids]['nodes']}
        new_Y = []
        for i in range(len(Y)):
            y = copy.deepcopy(Y[i])
            if Y[i]['score'] > math.log(thresholds):
                if is_valid({'start': Y[i]['start'], 'end': Y[i]['end']}, existed_edges, id2node):
                    existed_edges.append({'start': Y[i]['start'], 'end': Y[i]['end']})
                    y['predict'] = 1
                    if Y[i]['target'] == 1:
                        right += 1
                    else:
                        wrong += 1
                else:
                    y['predict'] = 0
            else:
                y['predict'] = 0
            y.pop('id')
            new_Y.append(y)
        p = 1.0 * right / (right + wrong + 1e-9)
        r = 1.0 * right / positive
        f1 = 2 * p * r / (p + r + 1e-9)
        print(index, p, r, f1)
        return right, wrong, positive, new_Y

    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_2020715/test/')
    test.load_all_datas()
    result = load_model_result(model.lower(), data_dir)
    right, wrong, total = 0, 0, 0
    for city in result:
        new_result = {}
        r_, w_, t_ = 0, 0, 0
        for index, v in result[city].items():
            y = []
            for sample in v:
                y.append({
                    'id': index,
                    'start': sample['start'],
                    'end': sample['end'],
                    'score': sample['score'],
                    'target': int(sample['target'])
                })
            y = sorted(y, key=lambda e: e['score'], reverse=True)
            r, w, t, y = metrics(y, index)
            r_ += r
            w_ += w
            t_ += t
            new_result[index] = y
        p = 1.0 * r_ / (r_ + w_ + 1e-9)
        r = 1.0 * r_ / t_
        f1 = 2 * p * r / (p + r + 1e-9)
        print(city, r_, w_, t_, p, r, f1)
        right += r_
        wrong += w_
        total += t_
        json.dump(new_result, open(data_dir + 'relational-gcn/final/Relational-GCN-' + city + '-result.json', 'w'), indent=2)
    p = 1.0 * right / (right + wrong + 1e-9)
    r = 1.0 * right / total
    f1 = 2 * p * r / (p + r + 1e-9)
    print(p, r, f1)


if __name__ == "__main__":
    data_dir = 'E:/python-workspace/CityRoadPrediction/data_2020715/'
    models = ['Relational-GCN']

    #roc(models)
    #precision_recall(models)
    #best_threshold('Relational-GCN')
    #predict('Relational-GCN')
