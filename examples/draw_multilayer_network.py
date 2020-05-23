'''
NOTE: this script requires pymnet, which is not included by default,
along with several other optional packages.
'''

import synthpops as sp
import numpy as np
import pymnet
import networkx as nx
import cmocean
import cmasher as cmr
import seaborn as sns
import covasim as cv
import sciris as sc

pop_size = 200
pop_type = 'synthpops'
undirected = True


# population = sp.make_population(n=200, generate=True)


contacts = dict(
    random = {'a':20},
    hybrid = {'h': 2.0, 's': 4, 'w': 6, 'c': 10},
    synthpops = {'h': 9.0, 's': 4, 'w': 6, 'c': 10},
    )

pars = {
    'pop_size': pop_size, # start with a small pool
    'pop_type': pop_type, # synthpops, hybrid
    'contacts': contacts[pop_type],
    'n_days': 1,
    # 'rand_seed': None,
}

# Create sim
sim = cv.Sim(pars=pars)
sim.initialize()

mnet = pymnet.MultilayerNetwork(aspects=1)


# fig = pl.figure(figsize=(16,16), dpi=120)
mapping = dict(a='All', h='Households', s='Schools', w='Work', c='Community')
colors = sc.vectocolor(sim.people.age, cmap='turbo')

keys = list(contacts[pop_type].keys())
keys.remove('c')
# nrowcol = np.ceil(np.sqrt(len(keys)))

G = nx.MultiGraph()

node_set = set()
home_set = set()
school_set = set()
work_set = set()

sample_size = 1

for p in range(sample_size):
    node_set.add(p)

    for i, layer in enumerate(keys):
        contacts = sim.people.contacts[layer]['p2'][sim.people.contacts[layer]['p1'] == p]
        node_set = node_set.union(contacts)

for p in node_set:
    for i, layer in enumerate(['h']):
        contacts = sim.people.contacts[layer]['p2'][sim.people.contacts[layer]['p1'] == p]
        node_set = node_set.union(contacts)

print(len(node_set))

G = nx.Graph()
G.add_nodes_from(node_set)
# A = pgv.AGraph()
# A.add_nodes_from(node_set)

# for i, layer in enumerate(keys):
for i, layer in enumerate(['h']):
    # print(layer)
    layer_name = mapping[layer]

    for p in node_set:
    # for p in range(sample_size):
#         node_set.add(p)
        contacts = sim.people.contacts[layer]['p2'][sim.people.contacts[layer]['p1'] == p]
#         print(p, len(contacts))
        for j in contacts:

            mnet[p, layer_name][j, layer_name] = 1

            if layer == 'h':
                G.add_edge(p, j)
                # A.add_edge(p, j)

#             node_set.add(j)

#             if layer == 'h':
#                 home_set.add(j)
#             elif layer == 's':
#                 school_set.add(j)
#             elif layer == 'w':
#                 work_set.add(j)
#         if len(contacts) > 0:
#             if layer == 'h':
#                 home_set.add(p)
#             elif layer == 's':
#                 school_set.add(p)
#             elif layer == 'w':
#                 work_set.add(p)

#     for p in node_set:
#         if p >= sample_size:
#             contacts = sim.people.contacts[layer]['p2'][sim.people.contacts[layer]['p1'] == p]
#             for j in contacts:

#                 mnet[p, layer_name][j, layer_name] = 1

# for p in node_set:
#     if p in home_set and p in school_set:
#         mnet[p, 'Households'][p, 'Schools'] = 1
#     if p in home_set and p in work_set:
#         mnet[p, 'Households'][p, 'Work'] = 1




# nn = mnet.iter_nodes('Households')
# for i in nn:
#     print(i)
# pos = nx.nx_pydot.graphviz_layout(G, prog='pydot')
# print(pos)

# print(A.layout())

import pydot

# P = nx.nx_pydot.to_pydot(G)
# print(P)
nx.nx_pydot.pydot_layout(G,prog='neato')


fig = pymnet.draw(mnet, defaultLayerAlpha=0.3, 
    layout='circular',
    # nodeCoords = pos,

    nodeLabelRule={}
    )
# fig.savefig('net.pdf')
fig.savefig('net_2.pdf')
