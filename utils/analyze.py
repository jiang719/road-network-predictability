import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def load_city_result(city, model, data_dir):
    return pickle.load(open(data_dir + model + '/result/' + city + '_result.pkl', 'rb'))


def load_model_result(model, data_dir):
    files = os.listdir(data_dir + model + '/result/')
    result = {}
    for file in files:
        city = file.split('_')[0].strip()
        #if city not in ['New york', 'Los angeles', 'Chicago', 'Houston', 'Philadelphia',
        #                'Phoenix', 'San diego', 'San antonio', 'Dallas', 'Detroit']:
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
    plt.title('ROC curve')
    plt.show()


def precision_recall(models):
    def metrics(Y):
        positive = sum([y['target'] for y in Y])

        thresholds = np.linspace(1, 1e-9, 1000)
        precision, recall = [], []
        index = 0
        right, wrong = 0, 0
        for _, th in enumerate(thresholds):
            for i in range(index, len(Y)):
                if Y[i]['score'] < math.log(th):
                    index = i
                    break
                if Y[i]['target'] == 1:
                    right += 1
                else:
                    wrong += 1
            precision.append(1.0 * right / (right + wrong + 1e-9))
            recall.append(1.0 * right / positive)
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
        print(len(y))
        y = sorted(y, key=lambda e: e['score'], reverse=True)
        precision, recall = metrics(y)

        plt.plot(recall, precision, label=model)
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()


if __name__ == "__main__":
    data_dir = 'E:/python-workspace/CityRoadPrediction/data_20200610/'
    models = ['Node2Vec', 'Struc2Vec', 'Graph-SAGE', 'Spectral-GCN', 'Relational-GCN', 'GAT']

    roc(models)
    precision_recall(models)

