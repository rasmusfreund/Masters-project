import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes with positions
positions = {
    'x2': (0, -1), 'x1': (0, 1),
    'h1': (2, 1.5), 'h2': (2, -1.5),
    'b1': (0, 2.5), 'b2': (2, 2.5),
    'o': (4, 0)
}

# Add nodes
G.add_node('x1', pos=positions['x1'])
G.add_node('x2', pos=positions['x2'])
G.add_node('h1', pos=positions['h1'])
G.add_node('h2', pos=positions['h2'])
G.add_node('o', pos=positions['o'])
G.add_node('b1', pos=positions['b1'])
G.add_node('b2', pos=positions['b2'])

# Add edges with weights
edges = [
    ('x1', 'h1', 'w_{11}'), ('x1', 'h2', 'w_{21}'),
    ('x2', 'h1', 'w_{12}'), ('x2', 'h2', 'w_{22}'),
    ('b1', 'h1', '-'), ('b1', 'h2', '-'),
    ('h1', 'o', 'w_{31}'), ('h2', 'o', 'w_{32}'),
    ('b2', 'o', '-')
]

G.add_weighted_edges_from([(u, v, 1.0) for u, v, _ in edges])  # weights are just placeholders for visualization

# Draw nodes
nx.draw_networkx_nodes(G, positions, node_size=2500, node_color='white', edgecolors='black')

# Draw edges
nx.draw_networkx_edges(G, positions, edgelist=G.edges, arrows=True)

# Draw labels for nodes
node_labels = {
    'x1': r'$x_1$', 'x2': r'$x_2$',
    'b1': r'$b_1$',
    'h1': r'$a_1$', 'h2': r'$a_2$',
    'b2': r'$b_2$',
    'o': r'$y$'
}
nx.draw_networkx_labels(G, positions, labels=node_labels, font_size=16)

# Draw labels for edges
edge_labels = {(u, v): f'${label}$' for u, v, label in edges if not 'b' in u}
nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=12)

# Additional text for biases
#plt.text(1.45, 3.1, r'$b_1$', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#plt.text(1.45, -1.25, r'$b_2$', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#plt.text(3.45, 0.95, r'$b_3$', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.title('Neural Network Structure with Variables', fontsize=16)
plt.axis('off')
plt.tight_layout()

save_path = "/mnt/c/Users/rasmu/Desktop/Bioinformatics/MSc/Thesis/img/nn_feedforward.png"
plt.savefig(save_path, dpi=300)
plt.show()
