import numpy as np
import json
import glob
import math

def load_graph_with_max_line_distance(file_name):
    nodes = json.load(open('../data_20200610/train/'+file_name+'nodes.json', 'r'))
    line_dis = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            point1 = (nodes[i]['lon'], nodes[i]['lat'])
            point2 = (nodes[j]['lon'], nodes[j]['lat'])
            line_dis[i,j]= math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return np.max(line_dis)


if __name__ == '__main__':
    file_list = glob.glob('../data_20200610/train/*edges.json')
    '''
    use in window
    '''
    name_list = []
    max_distance = 0
    for file_name in file_list:
        name_list.append((file_name.split('\\')[-1]).split('edges.json')[0])
    for city_name in name_list:
        dis = load_graph_with_max_line_distance(city_name)
        if dis>max_distance:
            max_distance = dis
    print('max distance in dataset :{}'.format(max_distance))