"""
Plot figure 4 of the Covasim supplementary file. Age mixing matrices on the left side and sample networks on the right side.
"""

import covasim as cv
import networkx as nx
import pylab as pl
import numpy as np
import sciris as sc
import synthpops as sp

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from collections import Counter
import pytest
import subprocess

import cmocean
import cmasher as cmr
import seaborn as sns

use_graphviz = True

graphviz_errormsg = f'Graphviz import failed, please install this first (conda install graphviz). If using Windows, ' \
           f'graphviz will fail on dependency neato. In this case you may want to set use_graphviz (line 27) to False to continue. ' \
           f'The figure will be produced using the spring layout algorithm and look quite than the example si_fig_4.png in the figures folder.'

# Try to import graphviz - may not be possible
try:
    import graphviz
except ImportError as E:
    errormsg = graphviz_errormsg
    raise ImportError(errormsg)

try:
    G = nx.DiGraph()
    pos = nx.nx_pydot.graphviz_layout(G)
except AssertionError as E:
    errormsg = graphviz_errormsg
    raise OSError(errormsg)


dir_path = os.path.dirname(os.path.realpath(__file__))

if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)

# try:
#     username = os.path.split(os.path.expanduser('~'))[-1]
#     fontdirdict = {
#         'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
#         'cliffk': '/home/cliffk/idm/covid-19/GoogleFonts',
#     }
#     if username not in fontdirdict:
#         fontdirdict[username] = os.path.expanduser(os.path.expanduser('~'), 'Dropbox', 'GoogleFonts')

#     font_path = fontdirdict[username]

#     fontpath = fontdirdict[username]
#     font_style = 'Roboto_Condensed'
#     fontstyle_path = os.path.join(fontpath, font_style, font_style.replace('_', '') + '-Light.ttf')
#     prop = font_manager.FontProperties(fname=fontstyle_path)
#     mplt.rcParams['font.family'] = prop.get_name()
# except:
#     mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16

fontsize = 34

### Draw networks ###

# Sample networks
pop_size = 180
pop_type = 'synthpops'
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'
undirected = True
n_days = 1
rand_seed = None
# with_facilities = True
with_facilities = False


pars = {
    'pop_size': pop_size,
    'pop_type': pop_type,
    'rand_seed': rand_seed,
    'n_days': n_days
}

sim = cv.Sim(pars)
popdict = cv.make_people(sim, generate=True, with_facilities=with_facilities, layer_mapping={'LTCF': 'l'})
sim = cv.Sim(pars, popfile=popdict, load_pop=True)

keys_to_plot = ['h', 'l', 's', 'w']
# keys_to_plot = ['h', 's', 'w']
keys = ['l', 'w', 's', 'h']

if with_facilities is False:
    if 'l' in keys_to_plot:
        keys_to_plot.remove('l')

    if 'l' in keys:
        keys.remove('l')

# vertical stacked on top of each other or columns of 2

left = 0.10
right = 0.8
top = 0.97
bottom = 0.07
wspace = 0.30  # horizontal spacing
hspace = 0.35  # vertical spacing

fig = plt.figure(figsize=(24, len(keys_to_plot) * 10))
axis_args = dict(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
plt.subplots_adjust(**axis_args)

mapping = dict(a='All', h='Households', s='Schools',
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
age_cutoffs = np.arange(0, 101, 10)  # np.array([0, 4, 6, 18, 22, 30, 45, 65, 80, 90, 100])
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
leg_left = right + 0.1
leg_right = 0.99
leg_bottom = bottom
leg_top = top + 0.01
leg_width = leg_right - leg_left
leg_height = leg_top - leg_bottom

ax_leg = fig.add_axes([leg_left, leg_bottom, leg_width, leg_height])
ax_leg.axis('off')
for age in age_cutoffs:
    nearest_age = sc.findnearest(sim.people.age, age)
    col = colors[nearest_age]
    if age != 100:
        ax_leg.plot(np.nan, np.nan, 'o', markersize=15, c=col, label=f'Age {age}-{age+9}')
    else:
        ax_leg.plot(np.nan, np.nan, 'o', markersize=15, c=col, label=f'Age {age}+')
ax_leg.legend(fontsize=fontsize + 4)


# Find indices of nodes
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
ndict = dict(h=60, s=160, w=160, l=160)
kdict = dict(h=0.7, s=1.0, w=2.0, l=1.5)
mdict = dict(h='^', s='o', w='s', l='D')
width = 0.2

mdict_2 = {i: mdict['h'] for i in idict['h']}
for i in idict['s']:
    mdict_2[i] = mdict['s']
for i in idict['w']:
    mdict_2[i] = mdict['w']

ndict_2 = {i: ndict['h'] for i in idict['h']}
for i in idict['s']:
    ndict_2[i] += 40
for i in idict['w']:
    ndict_2[i] += 40

# Set up 6 ax panel
ax = []
for i in range(len(keys_to_plot) * 2):
    ax.append(fig.add_subplot(len(keys_to_plot), 2, i+1))


layer_indices = {layer: l for l, layer in enumerate(keys_to_plot)}


for l, layer in enumerate(keys_to_plot):
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
    print(f'Drawing sample network layer: {layer}, with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    if use_graphviz:
        pos = nx.nx_pydot.graphviz_layout(G)
    else:
        pos = nx.spring_layout(G, k=kdict[layer], iterations=200)
    nx.draw(G, pos=pos, ax=ax[2*l + 1], node_size=ndict[layer],
            node_shape=mdict[layer],
            # node_shape=mdict_2,
            width=width, arrows=False, node_color=color)
    # still need to find a way to plot only the staff from LTCF layer in the households layer with the diamond marker
    if with_facilities:
        sublayers = 'hsw'
    else:
        sublayers = 'hsw'

    if layer == 'h':
        for sublayer in sublayers:
            sli = idict[sublayer]
            if sublayer == 's':
                sli = trimmed_s
            elif sublayer == 'h':
                sli = trimmed_h
            subG = G = nx.DiGraph()
            subG.add_nodes_from(sli)
            subpos = {i: pos[i] for i in sli}
            # sublayer_index = layer_indices[sublayer]
            sublayer_index = layer_indices['h']
            nx.draw(subG, pos=subpos, ax=ax[2*sublayer_index + 1], node_size=120,
                    node_shape=mdict[sublayer],
                    # node_shape=mdict_2,
                    width=width, arrows=False, node_color=colors[sli])
    ax[2*l + 1].set_title(mapping[layer], fontsize=fontsize + 10)


### Age mixing matrice ###

# create a large enough population to make the age mixing matrices
n = int(20e3)
generate = True
population = sp.make_population(n, generate=generate, with_facilities=with_facilities)

# aggregate age brackets
age_brackets = sp.get_census_age_brackets(sp.datadir, state_location=state_location, country_location=country_location)
age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

ages = []
for uid in population:
    ages.append(population[uid]['age'])
age_count = Counter(ages)
aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

matrix_dic = {}
density_or_frequency = 'density'
# density_or_frequency = 'frequency'

aggregate_flag = True
# aggregate_flag = False

logcolors_flag = True
logcolors_flag = False
# log color bounds

matrix_cmap = 'cmr.freeze_r'

vbounds = {}
if density_or_frequency == 'frequency':
    if aggregate_flag:
        vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e-0}
        vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
        vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
        vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}
    else:
        vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
        vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
        vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}
        vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}

elif density_or_frequency == 'density':
    if aggregate_flag:
        vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
        vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e1}
        vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e1}
        vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}

    else:
        vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
        vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
        vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
        vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}


im = []
cax = []
cbar = []
rotation = 66

for l, layer in enumerate(keys_to_plot):
    setting_code = layer.title()
    if setting_code == 'L':
        setting_code = 'LTCF'
    print(f'Plotting average age mixing contact matrix in layer: {layer}')
    matrix_dic[layer] = sp.calculate_contact_matrix(population, density_or_frequency, setting_code)

    if aggregate_flag:
        aggregate_matrix = sp.get_aggregate_matrix(matrix_dic[layer], age_by_brackets_dic)
        asymmetric_matrix = sp.get_asymmetric_matrix(aggregate_matrix, aggregate_age_count)

    else:
        asymmetric_matrix = sp.get_asymmetric_matrix(matrix_dic[layer], age_count)

    im.append(ax[2*l].imshow(asymmetric_matrix.T, origin='lower', interpolation='nearest', cmap=matrix_cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax'])))

    divider = make_axes_locatable(ax[2*l])
    cax.append(divider.new_horizontal(size="5%", pad=0.15))

    fig.add_axes(cax[l])
    cbar.append(fig.colorbar(im[l], cax=cax[l]))
    cbar[l].ax.tick_params(axis='y', labelsize=fontsize + 2)
    cbar[l].ax.set_ylabel(density_or_frequency.title() + ' of Contacts', fontsize=fontsize + 0)
    ax[2*l].tick_params(labelsize=fontsize + 2)
    ax[2*l].set_xlabel('Age', fontsize=fontsize + 2)
    ax[2*l].set_ylabel('Age of Contacts', fontsize=fontsize + 2)
    ax[2*l].set_title(mapping[layer], fontsize=fontsize + 10)

    if aggregate_flag:
        tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
        ax[2*l].set_xticks(np.arange(len(tick_labels)))
        ax[2*l].set_xticklabels(tick_labels, fontsize=fontsize, rotation=rotation)
        ax[2*l].set_yticks(np.arange(len(tick_labels)))
        ax[2*l].set_yticklabels(tick_labels, fontsize=fontsize + 2)
    else:
        ax[2*l].set_xticks(np.arange(0, len(age_count) + 1, 10))
        ax[2*l].set_yticks(np.arange(0, len(age_count) + 1, 10))


fig_path = os.path.join(dir_path, '..', 'figures')
fig_name = os.path.join(fig_path, 'si_fig_4b.pdf')

do_save = False
if do_save:
    fig.savefig(fig_name, format='pdf')
    fig.savefig(fig_name.replace('pdf', 'svg'), format='svg')
    fig.savefig(fig_name.replace('pdf', 'png'), format='png')
