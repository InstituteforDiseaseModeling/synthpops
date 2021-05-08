"""
Draw contact networks within different layers. Originally, draw_networks.py
"""

import covasim as cv
import networkx as nx
import pylab as pl
import numpy as np
import sciris as sc
import synthpops as sp

import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import cmocean
import cmasher as cmr
import seaborn as sns

import os

graphviz_errormsg = f'Graphviz import failed, please install this first (conda install graphviz). If using Windows, ' \
           f'graphviz will fail on dependency neato. In this case you may want to set use_graphviz (line 27) to False to continue. ' \
           f'The figure will be produced using the spring layout algorithm and look quite than the example si_fig_4.png in the figures folder.'

# Try to import graphviz - may not be possible
try:
    import graphviz
    G = nx.DiGraph()
    pos = nx.nx_pydot.graphviz_layout(G)
# except ImportError as E:
except:
    errormsg = graphviz_errormsg
    print(errormsg)


dir_path = os.path.dirname(os.path.realpath(__file__))

# Pretty fonts
username = os.path.expanduser('~')

fontdirdic = {
    'dmistry': os.path.join(username, 'Dropbox (IDM)', 'GoogleFonts'),
    'cliffk': os.path.join(username, 'idm', 'covid-19', 'GoogleFonts'),
}
if username not in fontdirdic:
    fontdirdic[username] = os.path.join(username, 'Dropbox', 'COVID-19', 'GoogleFonts')

try:
    fontpath = fontdirdic[username]
    fontstyle = 'Roboto_Condensed'
    fontstyle_path = os.path.join(fontpath, fontstyle, fontstyle.replace('_', '') + '-Light.ttf')
    mplt.rcParams['font.family'] = fontstyle.replace('_', ' ')
except:
    mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16

pop_size = 250
pop_type = 'synthpops'
location = 'seattle_metro'
undirected = True
n_days = 1
rand_seed = None
with_facilities = False

pars = {
    'pop_size': pop_size,
    'pop_type': pop_type,
    'rand_seed': rand_seed,
    'n_days': n_days,
}

sim = cv.Sim(pars)
popdict = cv.make_people(sim, generate=True, with_facilities=with_facilities, layer_mapping={'LTCF': 'l'})
sim = cv.Sim(pars, popfile=popdict, load_pop=True)

# Select which keys to plot!
# keys_to_plot = ['h', 'l']
# keys_to_plot = ['s', 'w']
keys_to_plot = ['h', 'l', 's', 'w']
# keys_to_plot = ['h', 's', 'w']

keys = ['l', 'w', 's', 'h']
# keys = ['w', 's', 'h']

if with_facilities is False:
    if 'l' in keys_to_plot:
        keys_to_plot.remove('l')

    if 'l' in keys:
        keys.remove('l')


# either vertical stacked on top of each other or columns of 2
# plot_stacked = True
plot_stacked = False


if plot_stacked:
    fig = plt.figure(figsize=(7, len(keys_to_plot) * 5))
    axis_args = dict(left=0.08, bottom=0.05, right=0.70, top=0.95, wspace=0.2, hspace=0.15)

elif len(keys_to_plot) % 2 != 0:
    fig = plt.figure(figsize=(7, len(keys_to_plot) * 5))
    axis_args = dict(left=0.08, bottom=0.05, right=0.70, top=0.95, wspace=0.2, hspace=0.15)

else:
    fig = plt.figure(figsize=(12, len(keys_to_plot)/2 * 5))
    axis_args = dict(left=0.08, bottom=0.05, right=0.80, top=0.95, wspace=0.2, hspace=0.15)


# axis_args = dict(left=0.10, bottom=0.05, right=0.70, top=0.95, wspace=0.25, hspace=0.15)
plt.subplots_adjust(**axis_args)

mapping = dict(a='All', h='Households',
               s='Schools',
               w='Workplaces', c='Community', l='Long Term Care Facilities')
discrete = False


# Stitch together two colormaps
cmap1 = plt.get_cmap('cmr.ember')
cmap2 = plt.get_cmap('cmr.lavender_r')
new_cmap_name = 'ember_lavender'

colors1 = cmap1(np.linspace(0.4, 1, 96))  # to truncate darker end of cmap1 change 0 to a value greater than 0, less than 1
colors2 = cmap2(np.linspace(0., 1, 128))  # to truncate darker end of cmap2 change 1 to a value less than 1, greater than 0

# transition_steps = 0  # heat+freeze
transition_steps = 4  # increase if closest ends of the color maps are far apart, values to try: 4, 8, 16
transition = mplt.colors.LinearSegmentedColormap.from_list("transition", [cmap1(1.), cmap2(0)])(np.linspace(0,1,transition_steps))
colors = np.vstack((colors1, transition, colors2))
colors = np.flipud(colors)

new_cmap = mplt.colors.LinearSegmentedColormap.from_list(new_cmap_name, colors)
cmap = new_cmap

# Assign colors to age groups
age_cutoffs = np.arange(0, 101, 10) # np.array([0, 4, 6, 18, 22, 30, 45, 65, 80, 90, 100])
if discrete:
    raw_colors = sc.vectocolor(len(age_cutoffs), cmap=cmap)
    colors = []
    for age in sim.people.age:
        ind = sc.findinds(age_cutoffs<=age)[-1]
        colors.append(raw_colors[ind])
    colors = np.array(colors)
else:
    age_map = sim.people.age*0.1+np.sqrt(sim.people.age)
    colors = sc.vectocolor(age_map, cmap=cmap)

# Create the legend
if plot_stacked:
    ax = fig.add_axes([0.85, 0.05, 0.14, 0.93])
elif len(keys_to_plot) % 2 != 0:
    ax = fig.add_axes([0.85, 0.05, 0.14, 0.93])
else:
    ax = fig.add_axes([0.82, 0.05, 0.14, 0.90])

ax.axis('off')
for age in age_cutoffs:
    nearest_age = sc.findnearest(sim.people.age, age)
    col = colors[nearest_age]
    if age != 100:
        plt.plot(np.nan, np.nan, 'o', c=col, label=f'Age {age}-{age+9}')
    else:
        plt.plot(np.nan, np.nan, 'o', c=col, label=f'Age {age}+')
plt.legend(fontsize=18)


# Find indices
idict = {}
hdfs = {}
for layer in keys:
    hdf = sim.people.contacts[layer].to_df()

    hdfs[layer] = hdf
    idict[layer] = list(set(list(hdf['p1'].unique()) + list(hdf['p2'].unique())))
    if layer == 'h':
        orig_h = idict[layer]
        idict[layer] = list(range(pop_size))

trimmed_s = sc.dcp(idict['s'])
for ind in idict['h']:
    if ind in trimmed_s and ind not in orig_h:
        trimmed_s.remove(ind)
trimmed_h = sc.dcp(idict['h'])
for ind in trimmed_s:
    if ind in trimmed_h:
        trimmed_h.remove(ind)
for ind in idict['w']:
    if ind in trimmed_h:
        trimmed_h.remove(ind)

# marker styles for people in different layers
ndict = dict(h=20, s=50, w=50, l=50)
kdict = dict(h=0.7, s=1.0, w=2.0, l=1.5)
mdict = dict(h='^', s='o', w='s', l='D')

use_graphviz = True

keys = keys_to_plot
for i, layer in enumerate(keys_to_plot):

    if plot_stacked:
        ax = plt.subplot(len(keys_to_plot), 1, i+1)

    elif len(keys_to_plot) % 2 != 0:
        ax = plt.subplot(len(keys_to_plot), 1, i+1)

    else:
        ax = plt.subplot(len(keys_to_plot)/2, 2, i+1)

    # ax = plt.subplot(len, 1, i+1)
    hdf = hdfs[layer]
    inds = idict[layer]
    color = colors[inds]

    G = nx.DiGraph()
    G.add_nodes_from(inds)
    p1 = hdf['p1']
    p2 = hdf['p2']
    G.add_edges_from(zip(p1, p2))
    if undirected:
        G.add_edges_from(zip(p2, p1))
    print(f'Layer: {layer}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}')

    if use_graphviz:
        pos = nx.nx_pydot.graphviz_layout(G)
    else:
        pos = nx.spring_layout(G, k=kdict[layer], iterations=200)
    nx.draw(G, pos=pos, ax=ax, node_size=ndict[layer], node_shape=mdict[layer], width=0.1, arrows=False, node_color=color)
    if layer == 'h':  # Warning, assumes that everyone is in a household
        for sublayer in 'hsw':
            sli = idict[sublayer]
            if sublayer == 's':
                sli = trimmed_s
            elif sublayer == 'h':
                sli = trimmed_h
            subG = G = nx.DiGraph()
            subG.add_nodes_from(sli)
            subpos = {i: pos[i] for i in sli}
            nx.draw(subG, pos=subpos, ax=ax, node_size=50, node_shape=mdict[sublayer], width=0.1, arrows=False, node_color=colors[sli])
    ax.set_title(mapping[layer], fontsize = 22)

# Save in the visualizations folder
fig_name = location + "_"
fig_name = fig_name + "".join([k for k in keys_to_plot])
# fig_name = fig_name + "_".join([mapping[k].replace(' ', '_').replace('_networks','') for k in keys_to_plot])
do_save = False
if do_save:
    os.makedirs('figures', exist_ok=True)
    fig.savefig(os.path.join('figures', fig_name + '_network_0.pdf'), format='pdf')
    fig.savefig(os.path.join('figures', fig_name + '_network_0.svg'), format='svg')
