import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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
shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

shortest_paths_df = pd.DataFrame(shortest_paths).fillna(float('inf'))

print("\nShortest Path Distances between all stations:")
print(shortest_paths_df)
