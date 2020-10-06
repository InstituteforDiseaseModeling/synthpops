import networkx as nx
import synthpops as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os


# Illustration of using a custom font
try:
    username = os.path.split(os.path.expanduser('~'))[-1]
    fontdirdict = {
        'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
    }
    if username not in fontdirdict:
        #  point to your GoogleFonts folder
        fontdirdict[username] = os.path.join(os.path.expanduser('~'), 'Dropbox', "COVID-19", 'GoogleFonts')

    font_path = fontdirdict[username]

    fontpath = fontdirdict[username]
    font_style = 'Roboto_Condensed'
    fontstyle_path = os.path.join(fontpath, font_style, font_style.replace('_', '') + '-Light.ttf')
    prop = font_manager.FontProperties(fname=fontstyle_path)
    mplt.rcParams['font.family'] = prop.get_name()
except:
    mplt.rcParams['font.family'] = 'Roboto'


# Main example code
if __name__ == '__main__':

    # directory for storing results
    do_run = True
    verbose = True
    do_save = False

    n = int(200)
    population = sp.make_population(n, generate=True)

    # Need to create contacts

    titles = {'h': 'Households', 's': 'Schools', 'w': 'Workplaces', 'c': 'Community'}

    fig = plt.figure(figsize=(16, 16))
    for i, layer in enumerate(['h', 's', 'w', 'c']):
        print(titles[layer])
        ax = plt.subplot(2, 2, i+1)

    #     hdf = sim.people.contacts[layer].to_df()

        G = nx.Graph()
        edges = set()
        if layer in ['h', 's', 'w']:

            nodes = population.keys()
            nodes = [uid for uid in nodes]
            if layer == 'H':
                G.add_nodes_from(nodes)

            for uid in nodes:
                # print(population[uid]['contacts'][layer.upper()])
                for cuid in population[uid]['contacts'][layer.upper()]:
                    edges.add((uid, cuid))

            G.add_edges_from(edges)

            if layer == 'S':
                for node in G.nodes():
                    if G.degee(node) == 0:
                        G.remove_node(node)

        else:
            p = 20./n
            G = nx.erdos_renyi_graph(n, p)

        print('Nodes:', G.number_of_nodes())
        print('Edges:', G.number_of_edges())

        nx.draw(G, ax=ax, node_size=1, width=0.05)
        ax.set_title(titles[layer], fontsize=24)

    if do_save:
        os.makedirs('figures', exist_ok=True)
        fig.savefig(os.path.join('figures', f"seattle_metro_{n}_contact_networks.png"))
    plt.show()
