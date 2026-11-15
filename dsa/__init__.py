"""
dsa – Data Structures and Algorithms package for AquaGrid.

Provides graph representations and traversal algorithms for network simulation.
"""

from dsa.graph import Graph, breadth_first_search, load_graph_from_db
from dsa.dijkstra import (
    DijkstraResult,
    dijkstra,
    find_shortest_path,
    create_demo_four_node_network,
)
from dsa.widest_path import (
    WidestPathResult,
    widest_path,
    find_widest_path,
)

__all__ = [
    "Graph",
    "breadth_first_search",
    "load_graph_from_db",
    "dijkstra",
    "find_shortest_path",
    "DijkstraResult",
    "create_demo_four_node_network",
    "WidestPathResult",
    "widest_path",
    "find_widest_path",
]
