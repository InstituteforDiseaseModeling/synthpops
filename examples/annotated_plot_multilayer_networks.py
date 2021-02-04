"""
Annotated example of plotting a multilayer network from a covasim sim object.
Using SynthPops to generate the multilayer network.
"""

import covasim as cv
import networkx as nx
import numpy as np
import sciris as sc
import os

# import synthpops only if you need to generate your own populations
# i.e. if you are not working directly with sim objects created and saved from covasim
import synthpops as sp

import matplotlib as mplt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# for additional colormaps
import cmocean as cmo
import cmasher as cmr
import seaborn as sns

mplt.rcParams['font.family'] = 'Roboto Condensed'
mplt.rcParams['font.size'] = 20

# Let's see if you import graphviz --- Windows machines seem to not be able to install this.
# Try a linux environment otherwise.
use_graphviz = True

graphviz_errormsg = f'Graphviz import failed, please install this first (conda install graphviz). If using Windows, ' \
           f'graphviz will fail on dependency neato. In this case you may want to set use_graphviz (line 27) to False to continue. ' \
           f'The figure will be produced using the spring layout algorithm.'

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


# dictionary to generate the sim and population in the sim

# if working directly with a covasim sim object then
# you can skip to line 125

pars = dict(
    n                               = .25e3,
    rand_seed                       = 123,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = 1,

    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    with_school_types               = 1,

    school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
    average_student_teacher_ratio   = 20,
    average_student_all_staff_ratio = 11,
    average_additional_staff_degree = 20,
    )

pars = sc.objdict(pars)

pop = sp.Pop(**pars)

popdict = pop.to_dict()

cvpopdict = cv.make_synthpop(population=sc.dcp(popdict), community_contacts=20)

school_ids = [None] * int(pars.n)
teacher_flag = [False] * int(pars.n)
staff_flag = [False] * int(pars.n)
student_flag = [False] * int(pars.n)
school_types = {'pk': [], 'es': [], 'ms': [], 'hs': [], 'uv': []}
school_type_by_person = [None] * int(pars.n)
schools = dict()

for uid, person in popdict.items():
    if person['scid'] is not None:
        school_ids[uid] = person['scid']
        school_type_by_person[uid] = person['sc_type']
        if person['scid'] not in school_types[person['sc_type']]:
            school_types[person['sc_type']].append(person['scid'])
        if person['scid'] in schools:
            schools[person['scid']].append(uid)
        else:
            schools[person['scid']] = [uid]
        if person['sc_teacher'] is not None:
            teacher_flag[uid] = True
        elif person['sc_student'] is not None:
            student_flag[uid] = True
        elif person['sc_staff'] is not None:
            staff_flag[uid] = True

assert sum(teacher_flag), 'Uh-oh, no teachers were found: as a school analysis this is treated as an error'
assert sum(student_flag), 'Uh-oh, no students were found: as a school analysis this is treated as an error'

people_pars = dict(
    pop_size = pars.n,
    beta_layer = {k: 1.0 for k in 'hwscl'},
    beta = 1.0,
)

people = cv.People(people_pars, strict=False, uid=cvpopdict['uid'], age=cvpopdict['age'], sex=cvpopdict['sex'],
                      contacts=cvpopdict['contacts'], school_id=np.array(school_ids),
                      schools=schools, school_types=school_types,
                      student_flag=student_flag, teacher_flag=teacher_flag,
                      staff_flag=staff_flag, school_type_by_person=school_type_by_person)

# # if working with a sim object from covasim:
# sim = cv.load('my-sim.sim')  # load it
# people = sim.people  # get the people

# what other attributes do people have? people.age for np.array of ages, people.uid for people ids
# people.contacts for a dictionary of contacts by layers with keys: 'h', 's', 'w', 'c', and 'l'

contact_layers_to_plot = ['h']
mapping = dict(a='All', h='Households', s='Schools', c='Community', w='Workplaces', l='Long Term Care Facilities')

cmap = plt.get_cmap('rocket')

discrete = False

# Assign colors to age groups
age_cutoffs = np.arange(0, 101, 10)  # np.array([0, 4, 6, 18, 22, 30, 45, 65, 80, 90, 100])
if discrete:
    raw_colors = sc.vectocolor(len(age_cutoffs), cmap=cmap)
    colors = []
    for age in people.age:
        ind = sc.findinds(age_cutoffs <= age)[-1]
        colors.append(raw_colors[ind])
    colors = np.array(colors)
else:
    age_map = people.age*0.1+np.sqrt(people.age)
    colors = sc.vectocolor(age_map, cmap=cmap)


# initialize a dictionary of the networks for each layer
graphs = sc.objdict()

# initialize a dictionary of the node positions / layout for each layer
positions = sc.objdict()

# dictionary mapping agent's age to color
node_colors = dict(zip(people.uid, colors))
layer_colors = [cmap(0.38), cmap(0.6)]


# loop through layers to plot and create the network for each
for layer in contact_layers_to_plot:
    hdf = people.contacts[layer].to_df()  # make pandas dataframe of the edges in the layer

    G = nx.Graph()
    G.add_nodes_from(people.uid)

    p1 = hdf['p1']  # get the first node in each edge as an array
    p2 = hdf['p2']  # get the second node in each edge as an array

    G.add_edges_from(zip(p1, p2))  # add edges to the network/graph object

    pos = nx.nx_pydot.graphviz_layout(G)  # get graph layout with graphviz

    # changing the layout --- if using your own layout, modify this or comment it out
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    if layer in ['s', 'w', 'l']:
        for p in pos:
            if G.degree(p) > 0:
                x, y = pos[p]
                if x < xmin:
                    xmin = x
                elif x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                elif y > ymax:
                    ymax = y
    else:
        for p in pos:
            x, y = pos[p]
            if x < xmin:
                xmin = x
            elif x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            elif y > ymax:
                ymax = y

    deltax = xmax - xmin
    deltay = ymax - ymin

    expansion_factor = {'x': {'w': 1./deltax, 's': 1./deltax, 'l': 1./deltax, 'h': 1./deltax},
                        'y': {'w': 1./deltay, 's': 1./deltay, 'l': 1./deltay, 'h': 1./deltay}}

    shift_factor = {'x': {'w': -xmin/deltax, 's': -xmin/deltax, 'l': -xmin/deltax, 'h': -xmin/deltax},
                    'y': {'w': -ymin/deltay, 's': -ymin/deltay, 'l': -ymin/deltay, 'h': -ymin/deltay}}

    if layer in expansion_factor['x']:
        for p in pos:
            x, y = pos[p]
            x *= expansion_factor['x'][layer]
            y *= expansion_factor['y'][layer]
            x += shift_factor['x'][layer]
            y += shift_factor['y'][layer]
            pos[p] = (x, y)

        positions[layer] = pos  # save updated positions of nodes for the layer
    positions[layer] = pos  # save positions of nodes for the layer
    graphs[layer] = G  # save the graph for the layer

width = 4.5
height = 4.5

fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=300, subplot_kw={'projection': '3d'}, tight_layout=True)
layer_names = {'h': 'Households', 's': 'Schools', 'w': 'Workplaces', 'l': 'Long Term\nCare Facilities'}

max_xlims = []
max_ylims = []

z_factor = 1.

for gi, layer in enumerate(graphs):
    G = graphs[layer]

    pos = positions[layer]

    # enumerate x, y, and z coordinates for every node as lists
    xs = list(list(zip(*list(pos.values())))[0])
    ys = list(list(zip(*list(pos.values())))[1])
    zs = [gi * z_factor] * len(xs)

    cs = colors  # list of node colors

    # get the edges to generate in each layer
    lines3d = np.array([(list(pos[i]) + [gi * z_factor], list(pos[j]) + [gi * z_factor]) for i, j in G.edges()])

    # create as 3d line collection
    line_collection = Line3DCollection(lines3d, color = '#666666', alpha =0.6, linewidth=1.)

    # add lines
    ax.add_collection3d(line_collection)

    # now add nodes
    ax.scatter(xs, ys, zs, s=8, c=cs, edgecolor=cs, marker='o', linewidth=0, alpha=1, zorder=100)

# angles to view from
angle = 0
height_angle = 90
ax.view_init(height_angle, angle)

# # how much do you want to zoom into the figure
ax.dist = 5.5

ax.set_axis_off()

fig_name = os.path.join('example_multilayer_networks_lwhs.pdf')
fig.savefig(fig_name, format='pdf')
fig.savefig(fig_name.replace('pdf', 'png'), format='png', dpi=500)
fig.savefig(fig_name.replace('pdf', 'svg'), format='svg')
