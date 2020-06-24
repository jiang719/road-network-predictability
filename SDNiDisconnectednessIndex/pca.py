import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import math
import glob

def nodes_to_list(nodes):
    new_nodes = []
    for n in nodes:
        new_nodes.append([n['osmid'],n['lon'],n['lat']])
    return new_nodes

def edges_to_dict(edges, sample=1):
    old_edges = {}
    for e in edges:
        if sample == 1:
            if e['start'] not in old_edges:
                old_edges[e['start']] = []
            old_edges[e['start']].append(e['end'])
        if sample == 2:
            if e['start'] not in old_edges:
                old_edges[e['start']] = []
            old_edges[e['start']].append(e['end'])
    return old_edges

def load_graph(file_name, sample=1):
    nodes = json.load(open('../data_20200610/train/'+file_name+'nodes.json', 'r'))
    edges = json.load(open('../data_20200610/train/'+file_name+'edges.json', 'r'))
    old_edges = edges_to_dict(edges, sample=sample)
    return nodes, old_edges

def visualization(nodeInfor, predictEdges, oldEdges, newEdges, city_name, SDNi):
    # step0: get the information
    nodeId = [nodeInfor[i][0] for i in range(len(nodeInfor))]
    longitude = [nodeInfor[i][1] for i in range(len(nodeInfor))]
    latitude = [nodeInfor[i][2] for i in range(len(nodeInfor))]

    # step1: generate the graph
    n = len(nodeId)
    A1 = np.array([[0] * n] * n)
    Graph1 = nx.Graph(A1)

    # step 2: label
    column = [str(nodeId[i]) for i in range(n)]
    mapping = {0: str(nodeId[0])}
    for i in range(0, len(column) - 1):
        mapping.setdefault(i + 1, column[i + 1])
    Graph1 = nx.relabel_nodes(Graph1, mapping)

    # step3: geolocation
    POS = list()
    for i in range(0, n):
        POS.append((float(longitude[i]), float(latitude[i])))
    for i in range(0, n):
        Graph1.nodes[column[i]]['pos'] = POS[i]

    num = 0
    # step 4: add edge
    for start in oldEdges:
        for end in oldEdges[start]:
            num = num + 1
            Graph1.add_edge(str(start), str(end), color='black', weight=1)
    # print('old num', num)
    for start in newEdges:
        for end in newEdges[start]:
            if (not (start in predictEdges and end in predictEdges[start])) and \
                    (not (end in predictEdges and start in predictEdges[end])):
                Graph1.add_edge(str(start), str(end), color='blue', weight=2)
    for start in predictEdges:
        for end in predictEdges[start]:
            if (start in newEdges and end in newEdges[start]) or \
                    (end in newEdges and start in newEdges[end]):
                Graph1.add_edge(str(start), str(end), color='green', weight=5)
            else:
                Graph1.add_edge(str(start), str(end), color='red', weight=2)

    edges = Graph1.edges()
    colors = [Graph1[u][v]['color'] for u, v in edges]
    weights = [Graph1[u][v]['weight'] for u, v in edges]
    # print(nx.cycle_basis(Graph1))
    plt.figure(1, figsize=(6, 6))
    plt.title('city: {}  SDNi: {:.3f}'.format(city_name, SDNi))
    nx.draw(Graph1, nx.get_node_attributes(Graph1, 'pos'), edge_color=colors, width=weights, node_size=10)#, with_labels = True)
    # plt.show()
    if SDNi>0:
        plt.savefig('figure/SDNi_{}_'.format(int(SDNi*100))+city_name+'png')
    else:
        plt.savefig('figure/SDNi_neg{}_'.format(int(-SDNi * 100)) + city_name + 'png')
    plt.clf()

def visualize(city_name, SDNi):
    sample = 1
    nodes, old_edges = load_graph(city_name, sample)
    visualization(nodes_to_list(nodes), dict(), old_edges, dict(), city_name, SDNi)


if __name__ == '__main__':
    with open('training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 7))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean)/data_std

    pca = PCA(n_components=3)
    newData = pca.fit_transform(data)
    SDNi = newData[:,0]

    print('min SDNi: ', np.min(newData[:,0]), ', max SDNi: ', np.max(newData[:,0]))
    print('explained_variance_ratio: ', pca.explained_variance_ratio_)
    print('explained_variance: ', pca.explained_variance_)

    # for select in np.linspace(0,1,51):
    #     # select = 0.98
    #     select_idx = int((len(city_index)-1)*select)
    #     select_SDNi_value = np.sort(SDNi)[select_idx]
    #     select_city_name = num_2_cityname[np.where(SDNi ==select_SDNi_value)[0][0]]
    #
    #     visualize(select_city_name, select_SDNi_value)