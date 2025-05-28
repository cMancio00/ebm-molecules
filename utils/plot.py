import numpy as np
import torch
import networkx as nx
from matplotlib import pyplot as plt

from utils.graph import DenseData
from sklearn.cluster import SpectralClustering


def plot_graph(graph: DenseData, ax=None, n_communities=1):
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()

    n_nodes = torch.sum(graph.mask).item()
    adj = graph.adj.numpy()[:n_nodes, :n_nodes]
    g = nx.from_numpy_array(adj)

    if n_communities > 1:
        sc = SpectralClustering(n_communities, affinity='precomputed', assign_labels='cluster_qr')
        sc.fit(adj)
        cluster_ids = sc.labels_
    else:
        cluster_ids = np.zeros(n_nodes)


    # build the position for the clusters
    pos = {}
    theta = (2 * np.pi) / n_communities
    r = 2
    centers = [(r * np.cos(i * theta), r * np.sin(i * theta)) for i in range(n_communities)]

    for i in range(n_communities):
        community_i = np.flatnonzero(cluster_ids == i)
        pos.update(nx.spring_layout(nx.induced_subgraph(g, community_i), seed=2222, center=centers[i]))

    # draw the nodes
    nx.draw_networkx_nodes(g, pos=pos, node_size=20, node_shape='o', cmap=plt.get_cmap('tab10'),
                           node_color=cluster_ids, edgecolors='k', ax=ax, vmin=0, vmax=10)

    # draw the edges
    edge_list = []
    alpha_list = []
    for u, v, weigth in g.edges.data('weight'):
        edge_list.append((u,v))
        alpha_list.append(weigth)

    nx.draw_networkx_edges(g, pos=pos, edgelist=edge_list, alpha=alpha_list, ax=ax)