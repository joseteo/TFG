import matplotlib.pyplot as plt
import networkx as nx


def draw_neural_network(layers):
    G = nx.DiGraph()

    # Add nodes
    for i, layer in enumerate(layers):
        for j in range(layer):
            G.add_node(f"L{i}N{j}", pos=(i, -j))

    # Add edges
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                G.add_edge(f"L{i}N{j}", f"L{i + 1}N{k}")

    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=1500, node_color='w', font_size=8, font_weight='bold',
            ax=ax)

    ax.set_title("Neural Network Architecture")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Neurons")

    plt.show()


# Define the architecture based on the provided network
layers = [1, 32, 64, 128, 10]  # Adjusted for the given DNN example
draw_neural_network(layers)
