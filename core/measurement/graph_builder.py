"""CervixAI V2 — Skeleton graph builder.

Converts a pruned binary skeleton into a weighted undirected graph
where:
- Nodes are skeleton pixel coordinates (y, x).
- Edges connect 8-adjacent skeleton pixels.
- Edge weights are Euclidean distances (1.0 for cardinal, √2 for diagonal).

Graph simplification: after building the full pixel-level graph,
junction pixels (degree ≥ 3) and endpoint pixels (degree 1) are used
as ''keypoints'' and the graph is compressed so edges become paths
between keypoints with their cumulative distance as weight.  This
significantly reduces the node count while preserving geodesic
distances, making Dijkstra efficient.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Type aliases
Coord = Tuple[int, int]       # (y, x)


class SkeletonGraphBuilder:
    """Build a simplified weighted graph from a pruned skeleton."""

    def build(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert a boolean skeleton array to a NetworkX graph.

        The returned graph is simplified: nodes are only endpoints
        and junctions; edges carry the geodesic distance of the path
        between them through skeleton pixels.

        Args:
            skeleton: Boolean (H, W) skeleton array.

        Returns:
            NetworkX undirected weighted graph.  Node labels are
            (y, x) integer tuples.  Edges carry attribute
            ``"weight"`` (float, in pixels).
        """
        pixel_graph = self._build_pixel_graph(skeleton)
        simplified = self._simplify(pixel_graph)
        logger.debug(
            "SkeletonGraphBuilder: %d nodes, %d edges (simplified)",
            simplified.number_of_nodes(),
            simplified.number_of_edges(),
        )
        return simplified

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pixel_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Build a pixel-level graph from the skeleton.

        Args:
            skeleton: Boolean skeleton array.

        Returns:
            NetworkX graph where every skeleton pixel is a node.
        """
        G: nx.Graph = nx.Graph()
        ys, xs = np.where(skeleton)
        coords: List[Coord] = list(zip(ys.tolist(), xs.tolist()))
        skel_set: Set[Coord] = set(coords)
        G.add_nodes_from(coords)

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        for (y, x) in coords:
            for dy, dx in offsets:
                nb: Coord = (y + dy, x + dx)
                if nb in skel_set:
                    w = math.sqrt(dy * dy + dx * dx)
                    G.add_edge((y, x), nb, weight=w)
        return G

    def _classify_nodes(self, G: nx.Graph) -> Tuple[Set[Coord], Set[Coord]]:
        """Identify endpoint and junction nodes.

        Args:
            G: Pixel-level skeleton graph.

        Returns:
            Tuple of (endpoints, junctions) as sets of (y, x) coords.
        """
        endpoints: Set[Coord] = set()
        junctions: Set[Coord] = set()
        for node in G.nodes():
            deg = G.degree(node)
            if deg == 1:
                endpoints.add(node)
            elif deg >= 3:
                junctions.add(node)
        return endpoints, junctions

    def _simplify(self, pixel_graph: nx.Graph) -> nx.Graph:
        """Compress pixel-level graph to keypoint-level graph.

        Traces paths between endpoints and junctions, collapsing each
        path to a single edge with the cumulative weight.

        Args:
            pixel_graph: Full pixel-level graph.

        Returns:
            Simplified graph with only keypoint nodes.
        """
        if pixel_graph.number_of_nodes() == 0:
            return nx.Graph()

        endpoints, junctions = self._classify_nodes(pixel_graph)
        keypoints: Set[Coord] = endpoints | junctions

        # If no junctions exist (simple path), use largest CC as-is
        if not keypoints:
            return pixel_graph

        simplified: nx.Graph = nx.Graph()
        simplified.add_nodes_from(keypoints)
        visited_edges: Set[Tuple[Coord, Coord]] = set()

        for start in keypoints:
            for path, dist in self._trace_paths(pixel_graph, start, keypoints):
                end = path[-1]
                edge_key = tuple(sorted([start, end]))
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)  # type: ignore[arg-type]
                if simplified.has_edge(start, end):
                    if simplified[start][end]["weight"] > dist:
                        simplified[start][end]["weight"] = dist
                else:
                    simplified.add_edge(start, end, weight=dist)

        return simplified

    def _trace_paths(
        self,
        G: nx.Graph,
        start: Coord,
        keypoints: Set[Coord],
    ) -> List[Tuple[List[Coord], float]]:
        """Trace all paths from ``start`` to the next keypoints.

        Args:
            G: Pixel-level graph.
            start: Starting keypoint node.
            keypoints: Set of all keypoint nodes.

        Returns:
            List of (path, total_distance) for each path found.
        """
        results = []
        for nb in G.neighbors(start):
            path, dist = self._walk(G, start, nb, keypoints)
            if path[-1] in keypoints and path[-1] != start:
                results.append((path, dist))
        return results

    def _walk(
        self,
        G: nx.Graph,
        prev: Coord,
        current: Coord,
        keypoints: Set[Coord],
    ) -> Tuple[List[Coord], float]:
        """Walk along the skeleton until a keypoint is reached.

        Args:
            G: Pixel-level graph.
            prev: Previous node (to avoid going back).
            current: Current node.
            keypoints: Stopping nodes.

        Returns:
            (path from current, cumulative distance).
        """
        path = [current]
        dist = G[prev][current]["weight"]

        while current not in keypoints or (current not in keypoints and len(path) == 1):
            if current in keypoints:
                break
            neighbours = [n for n in G.neighbors(current) if n != prev]
            if not neighbours:
                break
            nxt = neighbours[0]
            dist += G[current][nxt]["weight"]
            prev, current = current, nxt
            path.append(current)

        return path, dist
