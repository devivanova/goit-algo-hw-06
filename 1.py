import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import heapq

G = nx.Graph()

stations = ['A', 'B', 'C', 'D', 'E', 'F']
G.add_nodes_from(stations)

routes = [
    ('A', 'B', 7),
    ('A', 'C', 9),
    ('A', 'D', 14),
    ('B', 'C', 10),
    ('B', 'E', 15),
    ('C', 'D', 11),
    ('C', 'F', 2),
    ('D', 'F', 9),
    ('E', 'F', 6)
]


G.add_weighted_edges_from(routes)

# Визуализация графа
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=700,
        node_color='skyblue', font_size=12, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Transport Network of a City")
plt.show()

# Анализ основных характеристик графа
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degree_info = dict(G.degree())
average_degree = sum(degree_info.values()) / num_nodes

analysis = {
    "Number of Stations (Nodes)": num_nodes,
    "Number of Routes (Edges)": num_edges,
    "Average Degree": average_degree
}
degree_df = pd.DataFrame(list(degree_info.items()),
                         columns=["Station", "Degree"])

print("Main characteristics of the graph:")
for key, value in analysis.items():
    print(f"{key}: {value}")

print("\nDegrees of each station:")
print(degree_df)


# DFS
def dfs(graph, start):
    visited = set()
    stack = [start]
    dfs_path = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            dfs_path.append(node)
            stack.extend(reversed(list(graph[node])))

    return dfs_path


# BFS
def bfs(graph, start):
    visited = set()
    queue = [start]
    bfs_path = []

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            bfs_path.append(node)
            queue.extend(list(graph[node]))

    return bfs_path


dfs_path = dfs(G, 'A')
bfs_path = bfs(G, 'A')

print("DFS Path:", dfs_path)
print("BFS Path:", bfs_path)


# Алгоритм Дейкстры
def dijkstra(graph, start):

    shortest_distances = {node: float('inf') for node in graph.nodes}
    shortest_distances[start] = 0

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > shortest_distances[current_node]:
            continue

        for neighbor, data in graph[current_node].items():
            distance = data['weight']
            new_distance = current_distance + distance

            if new_distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return shortest_distances


custom_shortest_paths = {}
for station in G.nodes:
    custom_shortest_paths[station] = dijkstra(G, station)

custom_shortest_paths_df = pd.DataFrame(
    custom_shortest_paths).fillna(float('inf'))


print("\nShortest Path Distances between all stations:")
print(custom_shortest_paths_df)
