"""
Please note this examples takes a few minutes to run
"""

import networkx as nx
import matplotlib as mplt
import matplotlib.pyplot as plt
import covasim as cv

mplt.rcParams['font.family'] = 'Roboto'


if __name__ == '__main__':

    # directory for storing results
    do_run = True
    verbose = False
    do_save = False

    pars = {'pop_size': 1e3,
            'pop_type': 'synthpops',
            }

    sim = cv.Sim(pars)
    sim.initialize()

    titles = {'h': 'Households', 's': 'Schools', 'w': 'Workplaces', 'c': 'Community'}

    fig = plt.figure(figsize=(16, 16))
    for i, layer in enumerate(titles.keys()):
        print(titles[layer])
        ax = plt.subplot(2, 2, i+1)
        hdf = sim.people.contacts[layer].to_df()

        G = nx.Graph()

        if layer in ['h', 's', 'w']:
            G.add_nodes_from(set(list(hdf['p1'].unique()) + list(hdf['p2'].unique())))
            f = hdf['p1']
            t = hdf['p2']
            G.add_edges_from(zip(f, t))
        else:
            p = sim.pars['contacts']['c']/sim.pars['pop_size']
            G = nx.erdos_renyi_graph(sim.pars['pop_size'],p)

        print('Nodes:', G.number_of_nodes())
        print('Edges:', G.number_of_edges())

        nx.draw(G, ax=ax, node_size=1, width=0.05)
        ax.set_title(titles[layer], fontsize=24)

    if do_save:
        fig.savefig(f"seattle_metro_{sim.pars['pop_size']}_contact_networks_covasim.png")
    plt.show()
