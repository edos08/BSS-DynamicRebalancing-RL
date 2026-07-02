"""
Graph initialization and manipulation functions.
"""

import os
from typing import List, Optional, Tuple

import networkx as nx
import osmnx as ox
from geopy.distance import geodesic, great_circle
from tqdm import tqdm


def initialize_graph(
    places: list[str],
    network_type: str,
    graph_path: str,
    simplify_network: bool = False,
    remove_isolated_nodes: bool = False,
    nodes_to_remove: list[tuple[float, float]] | list[int] | None = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    save: bool = True,
) -> nx.MultiDiGraph:
    """
    Initialize the graph representing the road network.

    If the graph file exists, it will be loaded. Otherwise, it will be downloaded
    from OpenStreetMap and saved.

    Parameters:
        places: List of place names to download the road network data.
        network_type: Type of network to download ('bike', 'walk', 'drive', etc.).
        graph_path: Path to save/load the graph file.
        simplify_network: Whether to simplify the network by consolidating intersections.
        remove_isolated_nodes: Whether to remove isolated nodes from the network.
        nodes_to_remove: List of (lat, lon) coordinates of nodes to remove.
        bbox: Bounding box as (north, south, east, west) to truncate the graph.
        save: Whether to save the downloaded graph to the specified path.

    Returns:
        The graph representing the road network.
    """
    if os.path.isfile(graph_path):
        print("Network file already exists. Loading the network data...")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
    else:
        print("Network file does not exist. Downloading the network data...")

        # Ensure directory exists
        if save:
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)

        graph = ox.graph_from_place(places[0], network_type=network_type)

        if bbox is not None:
            # bbox input: (north, south, east, west)
            # OSMnx v2 expects: (left, bottom, right, top) = (west, south, east, north)
            north, south, east, west = bbox
            bbox_v2 = (west, south, east, north)
            print(f"Truncating graph to bounding box: {bbox_v2}")
            graph = ox.truncate.truncate_graph_bbox(G=graph, bbox=bbox_v2)
            graph = ox.truncate.largest_component(graph, strongly=False)

        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)

        # Download and compose graphs for additional places
        if len(places) > 1:
            for index in range(1, len(places)):
                grp = ox.graph_from_place(places[index], network_type=network_type)
                grp = ox.add_edge_speeds(grp)
                grp = ox.add_edge_travel_times(grp)
                graph = nx.compose(graph, grp)
                connect_disconnected_neighbors(graph, radius_meters=100)

        # Simplify the graph by consolidating intersections
        if simplify_network:
            G_proj = ox.project_graph(graph)
            G_cons = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=True)
            graph = ox.project_graph(G_cons, to_crs="epsg:4326")

        # Remove isolated nodes
        if remove_isolated_nodes:
            # Remove true isolates (degree 0) first
            graph.remove_nodes_from(list(nx.isolates(graph)))

            # Remove small disconnected fragments (e.g. single "solitary" nodes
            # left dangling after bbox truncation)
            if graph.number_of_nodes() > 0:
                components = list(nx.weakly_connected_components(graph))
                largest_component = max(components, key=len)
                nodes_to_drop = set(graph.nodes()) - largest_component
                if nodes_to_drop:
                    print(f"Removing {len(nodes_to_drop)} node(s) outside the main connected component...")
                    graph.remove_nodes_from(nodes_to_drop)

        # Remove specified nodes
        if nodes_to_remove is not None and len(nodes_to_remove) > 0:
            # check if instance of List[int] or List[Tuple[float, float]]
            if isinstance(nodes_to_remove[0], int):
                print("Removing specified nodes by ID...")
                for node in nodes_to_remove:
                    if node in graph:
                        graph.remove_node(node)
            else:
                for coord in nodes_to_remove:
                    nearest_node = ox.distance.nearest_nodes(graph, X=coord[1], Y=coord[0])
                    if nearest_node in graph:
                        graph.remove_node(nearest_node)

        if save:
            ox.save_graphml(graph, graph_path)
            print("Network data downloaded and saved successfully.")
        else:
            print("Network data downloaded successfully.")

    return graph


def load_graph(graph_path: str) -> nx.MultiDiGraph:
    """
    Load a graph from a GraphML file.

    Parameters:
        graph_path: Path to the GraphML file.

    Returns:
        The loaded graph.

    Raises:
        FileNotFoundError: If the graph file does not exist.
    """
    if os.path.isfile(graph_path):
        print("Loading network data...")
        graph = ox.load_graphml(graph_path)
        print("Network data loaded successfully.")
        return graph
    else:
        raise FileNotFoundError(f"Network file does not exist: {graph_path}")


def find_nearby_nodes(graph: nx.MultiDiGraph, target_node: int, radius_meters: float) -> List[int]:
    """
    Find nearby nodes within a specified radius around the target node.

    Parameters:
        graph: The graph representing the road network.
        target_node: The target node to find nearby nodes around.
        radius_meters: The radius in meters within which to find nearby nodes.

    Returns:
        A list of node IDs that are within the specified radius.

    Raises:
        ValueError: If the target node is not in the graph.
    """
    if target_node not in graph:
        raise ValueError(f"Node {target_node} is not in the graph.")

    target_coords = (graph.nodes[target_node]["y"], graph.nodes[target_node]["x"])
    nearby_nodes = []

    for node in graph.nodes:
        if node != target_node:
            node_coords = (graph.nodes[node]["y"], graph.nodes[node]["x"])
            dist = great_circle(target_coords, node_coords).meters

            if dist <= radius_meters:
                nearby_nodes.append(node)

    return nearby_nodes


def connect_disconnected_neighbors(graph: nx.MultiDiGraph, radius_meters: int) -> None:
    """
    Connect disconnected nodes in the graph by adding edges between them.

    Parameters:
        graph: The graph representing the road network (modified in place).
        radius_meters: The radius in meters within which to connect disconnected nodes.
    """
    tbar = tqdm(total=len(graph.nodes), desc="Connecting disconnected nodes")

    for node in graph.nodes:
        if "y" not in graph.nodes[node] or "x" not in graph.nodes[node]:
            print(f"Node {node} does not have valid coordinates.")
            continue

        nearby_nodes = find_nearby_nodes(graph, node, radius_meters)

        for neighbor in nearby_nodes:
            if not nx.has_path(graph, node, neighbor):
                node_coords = (graph.nodes[node]["y"], graph.nodes[node]["x"])
                neighbor_coords = (graph.nodes[neighbor]["y"], graph.nodes[neighbor]["x"])

                if "y" not in graph.nodes[neighbor] or "x" not in graph.nodes[neighbor]:
                    print(f"Neighbor {neighbor} does not have valid coordinates.")
                    continue

                distance_meters = great_circle(node_coords, neighbor_coords).meters
                speed_kph = 15.0
                travel_time_hours = distance_meters / 1000 / speed_kph
                weight = travel_time_hours * 3600

                graph.add_edge(node, neighbor, length=distance_meters, speed_kph=speed_kph, weight=weight)

        tbar.update(1)


def maximum_distance_between_points(G: nx.MultiDiGraph) -> float:
    """
    Compute the maximum distance between any two connected nodes in the graph.

    Parameters:
        G: The graph representing the road network.

    Returns:
        The maximum distance in meters between any two connected nodes.
    """
    max_distance = 0
    for u, v, data in G.edges(data=True):
        u_coords = (G.nodes[u]["y"], G.nodes[u]["x"])
        v_coords = (G.nodes[v]["y"], G.nodes[v]["x"])
        dist = geodesic(u_coords, v_coords).meters

        if dist > max_distance:
            max_distance = dist

    return max_distance


def is_within_graph_bounds(
    G: nx.MultiDiGraph,
    node_coords: Tuple[float, float],
    nearest_node: int,
    threshold: int = 500,
) -> bool:
    """
    Check if a point is within the bounds of the graph.

    Parameters:
        G: The graph representing the road network.
        node_coords: A tuple (lat, lon) of the point to check.
        nearest_node: The nearest node in the graph to the point.
        threshold: The maximum distance allowed between the point and the nearest node.

    Returns:
        True if the point is within the bounds of the graph, False otherwise.
    """
    nearest_node_coords = (G.nodes[nearest_node]["y"], G.nodes[nearest_node]["x"])
    distance_to_nearest_node = geodesic(node_coords, nearest_node_coords).meters
    return distance_to_nearest_node <= threshold
