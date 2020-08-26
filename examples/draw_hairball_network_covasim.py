"""
Please note this examples takes a few minutes to run
"""

import networkx as nx
import matplotlib as mplt
import matplotlib.pyplot as plt
import covasim as cv

mplt.rcParams['font.family'] = 'Roboto'


def create_sim(TRACE_PROB=None,  TEST_PROB=None, TRACE_TIME=None, TEST_DELAY=None):

    # initialize simulation
    sim = cv.Sim()

    # PARAMETERS
    pars = {'pop_size': 1e3}  # start with a small pool << actual population e4 # DEBUG

    # diagnosed individuals maintain same beta
    pars.update({
        'pop_type': 'synthpops',
        'beta_layer': {'h': 2.0, 's': 0, 'w': 0, 'c': 0},  # Turn off all but home
    })

    # update parameters
    sim.update_pars(pars=pars)

    return sim


if __name__ == '__main__':

    # directory for storing results
    do_run = True
    verbose = False
    do_save = False

    #fn = f'sim_sar.sim'
    sim = create_sim()
    sim.initialize()

    # Need to create contacts

    titles = {'h': 'Households', 's': 'Schools', 'w': 'Workplaces', 'c': 'Community'}

    fig = plt.figure(figsize=(16, 16))
    for i, layer in enumerate(['h', 's', 'w', 'c']):
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
