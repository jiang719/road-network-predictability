import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import math
import glob

def nodes_to_list(nodes):
    new_nodes = []
    for n in nodes:
        new_nodes.append([n['osmid'],n['lon'],n['lat']])
    return new_nodes

def nodes_to_dict(nodes):
    new_nodes = {}
    for n in nodes:
        new_nodes[n['osmid']] = (n['lon'], n['lat'])
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

def load_graph(file_name, train = True, sample=1):
    if train:
        nodes = json.load(open('../codeJiaweiXue_2020715_dataCollection/train/' + file_name + 'nodes.json', 'r'))
        edges = json.load(open('../codeJiaweiXue_2020715_dataCollection/train/' + file_name + 'edges.json', 'r'))
    else:
        nodes = json.load(open('../codeJiaweiXue_2020715_dataCollection/test/'+file_name+'nodes.json', 'r'))
        edges = json.load(open('../codeJiaweiXue_2020715_dataCollection/test/'+file_name+'edges.json', 'r'))
    old_edges = edges_to_dict(edges, sample=sample)
    return nodes, old_edges

def visualization(nodeInfor, predictEdges, oldEdges, newEdges):
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
    print("node number", len(Graph1.nodes))
    print("edge number", len(Graph1.edges))
    plt.figure(1, figsize=(6, 6))
    nx.draw(Graph1, nx.get_node_attributes(Graph1, 'pos'), edge_color=colors, width=weights, node_size=10)#, with_labels = True)
    plt.show()

def neg_avg_degree(old_edges):
    edges_degree_dict = {}
    for dict_name in old_edges:
        if dict_name not in edges_degree_dict:
            edges_degree_dict[dict_name] = 0
        for v in old_edges[dict_name]:
            if v not in edges_degree_dict:
                edges_degree_dict[v] = 0
    for dict_name in old_edges:
        for v in old_edges[dict_name]:
            edges_degree_dict[dict_name] +=1
            edges_degree_dict[v] += 1

    ## assign the large degree node with a fixed value 4
    for v in edges_degree_dict:
       if edges_degree_dict[v]>4:
           edges_degree_dict[v] = 4

    summary = np.zeros(len(edges_degree_dict))

    count = 0
    for idx, v in enumerate(edges_degree_dict):
        count += edges_degree_dict[v]
        summary[idx] = edges_degree_dict[v]
    pos_avg_count = count/len(edges_degree_dict)
    # print(summary)
    # print(np.sum(summary==1))
    frac_degree1 = np.sum(summary == 1) / len(edges_degree_dict)
    frac_degree2 = np.sum(summary == 2) / len(edges_degree_dict)
    frac_degree3 = np.sum(summary == 3) / len(edges_degree_dict)
    frac_degree4 = np.sum(summary == 4) / len(edges_degree_dict)
    return pos_avg_count, frac_degree1, frac_degree2, frac_degree3, frac_degree4

def shortest_path_distance(nodes, old_edges):
    nodes_dict = nodes_to_dict(nodes)
    nodes_list = [v for v in nodes_dict]
    nodes_id_to_idx = {}
    for i, v in enumerate(nodes_list):
        nodes_id_to_idx[v] = i

    distances = {}
    for node_name in old_edges:
        if node_name not in distances:
            distances[node_name] = {}
        for node_name_neighbour in old_edges[node_name]:
            if node_name_neighbour not in distances:
                distances[node_name_neighbour] = {}
            point1 = nodes_dict[node_name]
            point2 = nodes_dict[node_name_neighbour]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            distances[node_name][node_name_neighbour] = distance
            distances[node_name_neighbour][node_name] = distance

    shortest_path_distance_matrix = np.zeros((len(nodes_list), len(nodes_list)))
    for i in range(len(nodes_list)):
        shortest_distance = DjikstraAlg(nodes_list, nodes_list[i], distances)
        for node_name in shortest_distance:
            shortest_path_distance_matrix[i, nodes_id_to_idx[node_name]] = shortest_distance[node_name]

    straight_line_distance_matrix = np.zeros((len(nodes_list), len(nodes_list)))
    for i in range(len(nodes_list)):
        for j in range(len(nodes_list)):
            point1 = nodes_dict[nodes_list[i]]
            point2 = nodes_dict[nodes_list[j]]
            straight_line_distance_matrix[i,j] = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    return shortest_path_distance_matrix, straight_line_distance_matrix

def DjikstraAlg(nodes, current_node, distances):
    unvisited = {node: float('inf') for node in nodes}  # using None as +inf
    visited = {}

    current = current_node
    currentDistance = 0
    unvisited[current] = currentDistance

    while True:
        for neighbour, distance in distances[current].items():
            if neighbour not in unvisited: continue
            newDistance = currentDistance + distance
            if unvisited[neighbour] is float('inf') or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]

    return visited

def LogCircuity(nodes, old_edges, r1, r2):
    shortest_path_distance_matrix, straight_line_distance_matrix = shortest_path_distance(nodes, old_edges)
    between_r1_r2 = np.logical_and(straight_line_distance_matrix<r2, straight_line_distance_matrix>r1)
    sum_straight_line = np.sum(straight_line_distance_matrix * between_r1_r2)
    sum_network_path = np.sum(shortest_path_distance_matrix * between_r1_r2)
    # print(sum_network_path, sum_straight_line)
    # print(np.log10(sum_network_path) - np.log10(sum_straight_line))
    if sum_network_path ==0 or sum_straight_line ==0:
        return 0
    else:
        return np.log10(sum_network_path) - np.log10(sum_straight_line)

def Bridge(nodes, old_edges):
    from Bridges import Graph
    graph = Graph(len(nodes))
    nodes_dict = nodes_to_dict(nodes)
    nodes_list = [v for v in nodes_dict]
    nodes_id_to_idx = {}
    for i, v in enumerate(nodes_list):
        nodes_id_to_idx[v] = i

    ######### find the total edges
    edges_count = 0
    for node_name in old_edges:
        for node_name_neighbour in old_edges[node_name]:
            graph.addEdge(nodes_id_to_idx[node_name], nodes_id_to_idx[node_name_neighbour])
            edges_count += 1

    ######## find all bridges and dead-ends
    graph.bridge()
    bridge_bool = np.zeros(len(graph.bridges))
    ### classify the node as bridge or dead-end (has a node with degree 1)
    edges_degree_dict = {}
    for dict_name in old_edges:
        if dict_name not in edges_degree_dict:
            edges_degree_dict[dict_name] = 0
        for v in old_edges[dict_name]:
            if v not in edges_degree_dict:
                edges_degree_dict[v] = 0
    for dict_name in old_edges:
        for v in old_edges[dict_name]:
            edges_degree_dict[dict_name] += 1
            edges_degree_dict[v] += 1

    for idx, edge in enumerate(graph.bridges):
        # print(nodes_list[edge[0]], edges_degree_dict[nodes_list[edge[0]]])
        if edges_degree_dict[nodes_list[edge[0]]] == 1 or edges_degree_dict[nodes_list[edge[1]]] == 1:
            bridge_bool[idx] = 0
        else:
            bridge_bool[idx] = 1

    frac_edge_bridges = sum(bridge_bool)/float(edges_count)
    frac_edge_deadends = sum(1 - bridge_bool)/float(edges_count)

    ####### find all edges length
    length_sum = 0
    for node_name in old_edges:
        for node_name_neighbour in old_edges[node_name]:
            point1 = nodes_dict[node_name]
            point2 = nodes_dict[node_name_neighbour]
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            length_sum =  length_sum + distance

    length_sum_bridge = 0
    length_sum_deadend = 0

    for idx, edge in enumerate(graph.bridges):
        point1 = nodes_dict[nodes_list[edge[0]]]
        point2 = nodes_dict[nodes_list[edge[1]]]
        distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        if bridge_bool[idx] == 1:
            length_sum_bridge = length_sum_bridge + distance
        else:
            length_sum_deadend = length_sum_deadend + distance

    frac_length_bridges = length_sum_bridge / length_sum
    frac_length_deadend = length_sum_deadend / length_sum
    # for idx, edge in enumerate(graph.bridges):
    #     print(nodes_list[edge[0]], nodes_list[edge[1]], bridge_bool[idx])
    # print('hehe', graph.bridges)
    # for v in graph.bridges:
    #     print(nodes_list[v[0]], nodes_list[v[1]])
    # return len(graph.bridges)/float(edges_count)
    return frac_edge_bridges, frac_edge_deadends, frac_length_bridges, frac_length_deadend

def index_city(city_name, train, visualize = False):
    sample = 1
    nodes, old_edges = load_graph(city_name, train, sample)
    '''
    check the uniqueness expression
    '''
    index_dict = {}

    # ############ neg_avg_degree
    neg_avg_count, frac_degree1, frac_degree2, frac_degree3, frac_degree4 = neg_avg_degree(old_edges)
    index_dict['neg_degree'] = neg_avg_count
    index_dict['frac_degree1'] = frac_degree1
    index_dict['frac_degree2'] = frac_degree2
    index_dict['frac_degree3'] = frac_degree3
    index_dict['frac_degree4'] = frac_degree4

    index_dict['log_circuity_0_0p005'] = LogCircuity(nodes, old_edges, 0.0, 0.005)
    index_dict['log_circuity_0p005_0p02'] = LogCircuity(nodes, old_edges, 0.005, 0.02)

    frac_edge_bridges, frac_edge_deadends, frac_length_bridges, frac_length_deadend = Bridge(nodes, old_edges)

    index_dict['frac_edge_bridges'] = frac_edge_bridges
    index_dict['frac_edge_deadends'] = frac_edge_deadends
    index_dict['frac_length_bridges'] = frac_length_bridges
    index_dict['frac_length_deadend'] = frac_length_deadend

    # print(index_dict)
    #
    # with open('data.txt', 'w') as outfile:
    #     json.dump(index_dict, outfile)

    # ############
    # LogCircuity(nodes, old_edges, 0.003, 0.006)
    # visualize the graph
    if visualize:
        visualization(nodes_to_list(nodes), dict(), old_edges, dict())
    return index_dict

if __name__ == '__main__':
    # city_name = 'Wichita10'#'Arlington0'#'Arlington0'##''Albuquerque7'#'Chicago0'
    # print(index_city(city_name, True))
    train = True
    if train:
        file_list = glob.glob('../codeJiaweiXue_2020715_dataCollection/train/*edges.json')
    else:
        file_list = glob.glob('../codeJiaweiXue_2020715_dataCollection/test/*edges.json')
    '''
    use in window
    '''
    name_list = []
    max_distance = 0
    for file_name in file_list:
        name_list.append((file_name.split('\\')[-1]).split('edges.json')[0])

    all_city_index = {}
    for idx, city in enumerate(name_list):
        print('{}/{} city: {}'.format(idx, len(name_list), city))
        all_city_index[city] = index_city(city, train = train)
    # all_city_index[name_list[0]] = index_city(name_list[0])
    # all_city_index[name_list[1]] = index_city(name_list[1])
    # print(all_city_index)
    if train:
        with open('training_set_index.txt', 'w') as outfile:
            json.dump(all_city_index, outfile)
    else:
        with open('test_set_index.txt', 'w') as outfile:
            json.dump(all_city_index, outfile)

