import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define central node
central_node = "Deep Learning Unit 1"

# Main branches (blue nodes)
main_branches = [
    "Deep Learning",
    "Machine Learning Basics",
    "Tensors",
    "Computation Graph",
    "Variables & Operations",
    "DL Libraries",
    "Optimizers",
    "Vanishing Gradient",
    "TensorFlow Intro",
    "TensorBoard"
]

# Sub-nodes (green nodes) connected to main branches
sub_nodes = {
    "Deep Learning": ["Neurons", "Layers", "Rules from data", "Feature Hierarchy", "Embeddings", "dtype"],
    "Machine Learning Basics": ["PyTorch", "Auto Differentiation", "ReLU", "BatchNorm", "Rank"],
    "Tensors": ["tf.constant", "Shape", "Rank", "tf.Variable", "ResNet", "Cat vs Dog Example"],
    "Computation Graph": ["Nodes=Ops", "Edges=Tensors", "Build", "Histograms", "Sigmoid issue"],
    "Variables & Operations": ["add/multiply/matmul", "Graph", "Train", "Loss/Accuracy", "Deploy"],
    "DL Libraries": ["TensorFlow", "Keras", "RMSprop", "Adam (default)", "SGD"],
    "Optimizers": ["Adam (default)", "RMSprop", "SGD", "Keras", "tf.Variable", "Shape"],
    "Vanishing Gradient": ["ReLU", "BatchNorm", "tf.constant", "ResNet", "Cat vs Dog Example"],
    "TensorFlow Intro": ["TensorBoard", "Train", "Graph", "add/multiply/matmul"],
    "TensorBoard": ["Loss/Accuracy", "Deploy", "Graph", "add/multiply/matmul"]
}

# Add central node
G.add_node(central_node, color='gold', size=800)

# Add main branches
for branch in main_branches:
    G.add_node(branch, color='skyblue', size=600)

# Add sub-nodes
for branch, nodes in sub_nodes.items():
    for node in nodes:
        G.add_node(node, color='lightgreen', size=400)

# Connect central node to main branches
for branch in main_branches:
    G.add_edge(central_node, branch)

# Connect main branches to their sub-nodes
for branch, nodes in sub_nodes.items():
    for node in nodes:
        G.add_edge(branch, node)

# Set up the layout
pos = nx.spring_layout(G, k=2, iterations=50)

# Node positions and sizes
node_colors = [G.nodes[node].get('color', 'lightgray') for node in G.nodes()]
node_sizes = [G.nodes[node].get('size', 300) for node in G.nodes()]

# Draw the graph
plt.figure(figsize=(20, 16), dpi=100)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='black')
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, alpha=0.7, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_family='sans-serif')

# Title
plt.title("Deep Learning Unit 1 - Mindmap", fontsize=18, fontweight='bold', pad=40)

# Remove axis
plt.axis('off')

# Adjust layout to prevent label overlap
plt.tight_layout()

# Show plot
plt.show()