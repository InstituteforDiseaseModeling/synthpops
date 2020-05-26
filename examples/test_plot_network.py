import numpy as np
import networkx as nx
import matplotlib as mplt
import matplotlib.pyplot as plt




def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata=[]; gdata=[]; bdata=[]
    for istep in range(nsteps):
        r = rgbdata[istep,0]
        g = rgbdata[istep,1]
        b = rgbdata[istep,2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red':   rdata,
                 'green': gdata,
                 'blue':  bdata}

    return mpl_data



if __name__ == '__main__':

    N = 1000
    p = 0.15

    # G = nx.erdos_renyi_graph(N,p)
    G = nx.random_geometric_graph(N,p)

    # bb = nx.betweenness_centrality(G)
    # nx.set_node_attributes(G, bb, 'betweenness')

    # nx.set_node_attributes(G, [bb,bb], 'pos')

    pos = nx.get_node_attributes(G,'pos')
    print(pos)

    dmin = 1
    ncenter = 0
    for n in pos:
        x,y = pos[n]
        print(x,y)
        d = (x-0.5)**2 + (y-0.5)**2
        if d < dmin:
            ncenter = n
            dmin = d

    path_len = nx.single_source_shortest_path_length(G,ncenter)

    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(G,pos,nodelist = [ncenter], alpha = 0.2, lw = 0.2)
    nx.draw_networkx_nodes(G,pos,nodelist = [k for k in path_len.keys()],
                                node_size = 20,
                                node_color = [v for v in path_len.values()],
                                # node_color = np.arange(N),
                                # cmap = mplt.cm.get_cmap('turbo')
                                )

    plt.show()


