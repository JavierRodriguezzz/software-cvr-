import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class CenterlineExtractor:
    def extract(self, skeleton: np.ndarray) -> np.ndarray:
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 2:
            return np.zeros((0, 2), dtype=np.intp)
        coords = list(zip(ys.tolist(), xs.tolist()))
        idx_of = {c: i for i, c in enumerate(coords)}
        skel_set = set(coords)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for y, x in coords:
            i = idx_of[(y, x)]
            for dy, dx in offsets:
                neighbor = (y + dy, x + dx)
                if neighbor in skel_set:
                    j = idx_of[neighbor]
                    w = (dy ** 2 + dx ** 2) ** 0.5
                    G.add_edge(i, j, weight=w)
        if G.number_of_nodes() == 0:
            return np.zeros((0, 2), dtype=np.intp)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        start = next(iter(G.nodes))
        lengths_start = nx.single_source_dijkstra_path_length(G, start, weight="weight")
        node_a = max(lengths_start, key=lengths_start.get)
        lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
        node_b = max(lengths_a, key=lengths_a.get)
        path = nx.shortest_path(G, node_a, node_b, weight="weight")
        centerline = np.array([coords[i] for i in path])
        return centerline
