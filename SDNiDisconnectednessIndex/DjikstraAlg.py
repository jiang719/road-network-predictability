'''
adapted from https://stackoverflow.com/questions/22897209/dijkstras-algorithm-in-python
'''
# nodes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I')
# distances = {
#     'B': {'A': 5, 'D': 1, 'G': 2},
#     'A': {'B': 5, 'D': 3, 'E': 12, 'F' :5},
#     'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
#     'G': {'B': 2, 'D': 1, 'C': 2},
#     'C': {'G': 2, 'E': 1, 'F': 16},
#     'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
#     'F': {'A': 5, 'E': 2, 'C': 16},
#     'H': {'I':1},
#     'I': {'H':1}}
#
# unvisited = {node: None for node in nodes} #using None as +inf
#
# visited = {}
# current = 'B'
# currentDistance = 0
# unvisited[current] = currentDistance
# print(unvisited)
# while True:
#     for neighbour, distance in distances[current].items():
#         if neighbour not in unvisited: continue
#         newDistance = currentDistance + distance
#         if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
#             unvisited[neighbour] = newDistance
#     visited[current] = currentDistance
#     print(unvisited)
#     del unvisited[current]
#     if not unvisited: break
#     candidates = [node for node in unvisited.items() if node[1]]
#     print(candidates)
#     current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]
#
# print(visited)


nodes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I')
distances = {
    'B': {'A': 5, 'D': 1, 'G': 2},
    'A': {'B': 5, 'D': 3, 'E': 12, 'F' :5},
    'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
    'G': {'B': 2, 'D': 1, 'C': 2},
    'C': {'G': 2, 'E': 1, 'F': 16},
    'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
    'F': {'A': 5, 'E': 2, 'C': 16},
    'H': {'I':1},
    'I': {'H':1}}

unvisited = {node: float('inf') for node in nodes} #using None as +inf

visited = {}
current = 'B'
currentDistance = 0
unvisited[current] = currentDistance
print(unvisited)
while True:
    for neighbour, distance in distances[current].items():
        if neighbour not in unvisited: continue
        newDistance = currentDistance + distance
        if unvisited[neighbour] is float('inf') or unvisited[neighbour] > newDistance:
            unvisited[neighbour] = newDistance
    visited[current] = currentDistance
    print(unvisited)
    del unvisited[current]
    if not unvisited: break
    candidates = [node for node in unvisited.items() if node[1]]
    print(candidates)
    current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]

print(visited)