from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np

# Initialize our Q matrix
Q = defaultdict(int)

# Create k-means traders by volume
volume = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]).reshape(-1, 1)

# Apply k-means clustering
kmeans = KMeans(n_clusters=9, random_state=0).fit(volume)

# Create a graph based on k-means traders
G = nx.Graph()

# Add nodes to the graph for each trader
for i, label in enumerate(kmeans.labels_):
    G.add_node(i)

# Add edges between nodes based on their distance in the k-means clustering
for i in range(len(kmeans.labels_)):
    for j in range(i+1, len(kmeans.labels_)):
        if kmeans.labels_[i] == kmeans.labels_[j]:
            G.add_edge(i, j)

# Update Q matrix for every edge in the graph
for i, j in G.edges:
    Q[(i,i)]+= -1
    Q[(j,j)]+= -1
    Q[(i,j)]+= 2

# Set up QPU parameters
chainstrength = 8
numruns = 10  # Reduced number of runs

# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')

# Display results
print("QPU response:")
print(response)
