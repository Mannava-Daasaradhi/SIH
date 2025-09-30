import networkx as nx
import matplotlib.pyplot as plt

def create_railway_network():
    """
    Creates a simple railway network graph using networkx.

    Layout represents a mainline with a siding loop.
    - S_A: Main Station A
    - S_B: Main Station B
    - J_A: Junction before the siding
    - J_B: Junction after the siding
    - ML: A point on the Main Line between junctions
    - SL: A point on the Siding Loop between junctions
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Define the nodes (stations, junctions, track points)
    nodes = {
        "S_A": {"type": "station"},
        "J_A": {"type": "junction"},
        "ML": {"type": "track_point"},
        "SL": {"type": "siding_point"},
        "J_B": {"type": "junction"},
        "S_B": {"type": "station"}
    }
    
    # Add nodes with their attributes to the graph
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # Define the edges (track segments) with distances
    # Tuple format: (start_node, end_node, distance_in_km)
    edges_with_distances = [
        # Path from Station A to Station B via Main Line
        ("S_A", "J_A", 5),
        ("J_A", "ML", 10),
        ("ML", "J_B", 10),
        ("J_B", "S_B", 5),

        # The siding loop path
        ("J_A", "SL", 4), # Siding is usually shorter/slower
        ("SL", "J_B", 4),

        # Return paths (for trains going from B to A)
        ("S_B", "J_B", 5),
        ("J_B", "ML", 10),
        ("ML", "J_A", 10),
        ("J_A", "S_A", 5),
        
        # Siding return path
        ("J_B", "SL", 4),
        ("SL", "J_A", 4),
    ]

    # Add edges to the graph with distance as a weight attribute
    for u, v, dist in edges_with_distances:
        G.add_edge(u, v, weight=dist)
        
    print("Railway network created successfully!")
    print(f"Nodes: {G.nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    
    return G

def visualize_network(G):
    """
    Creates and displays a visualization of the railway network graph.
    """
    # Position nodes for a clear layout
    pos = {
        "S_A": (0, 0), "J_A": (1, 0), "ML": (2, 0.5), "SL": (2, -0.5),
        "J_B": (3, 0), "S_B": (4, 0)
    }

    # Define colors for different node types
    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['type'] == 'station':
            node_colors.append('skyblue')
        elif node[1]['type'] == 'junction':
            node_colors.append('salmon')
        else:
            node_colors.append('lightgreen')
            
    plt.figure(figsize=(12, 6))
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors, 
            font_size=10, font_weight='bold', arrowsize=20)
    
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Simplified Railway Network Visualization")
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    rail_network = create_railway_network()
    visualize_network(rail_network)