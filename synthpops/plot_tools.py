import sciris as sc
import numpy as np
import networkx as nx
from . import synthpops as sp
from .config import datadir

import copy
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cmocean
import functools
import os

def calculate_contact_matrix(contacts,density_or_frequency='density'):
    uids = contacts.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    F_dic = {}
    for k in ['M','H','S','W','R']:
        F_dic[k] = np.zeros((num_ages,num_ages))

    for n,uid in enumerate(uids):
        layers = contacts[uid]['contacts']
        age = contacts[uid]['age']
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            contact_ages = np.array([int(a) for a in contact_ages])

            if len(contact_ages) > 0:
                if density_or_frequency == 'density':
                    for ca in contact_ages:
                        F_dic[k][age,ca] += 1.0/len(contact_ages)
                    # F_dic[k][age, contact_ages] += 1 / len(contact_ages)
                elif density_or_frequency == 'frequency':
                    for ca in contact_ages:
                        F_dic[k][age,ca] += 1.0
                    # F_dic[k][age, contact_ages] += 1

    return F_dic


def plot_contact_frequency(freq_matrix_dic,setting_code,age_count,aggregate_age_count,age_brackets,age_by_brackets_dic,density_or_frequency='density',logcolors_flag=False,aggregate_flag=True):

    print(setting_code)
    cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
    # cmap = mplt.cm.get_cmap(cmocean.cm.matter_r)

    fig = plt.figure(figsize = (9,9))
    ax = []
    cax = []
    cbar = []
    leg = []
    implot = []


    titles = {'H': 'Household', 'S': 'School', 'W': 'Work'}

    if aggregate_flag:
        num_agebrackets = len(set(age_by_brackets_dic.values()))
        aggregate_M = sp.get_aggregate_matrix(freq_matrix_dic[setting_code],age_by_brackets_dic,num_agebrackets)
        asymmetric_M = sp.get_asymmetric_matrix(aggregate_M,aggregate_age_count)
    else:
        num_agebrackets = len(age_brackets)
        asymmetric_M = sp.get_asymmetric_matrix(freq_matrix_dic[setting_code],age_count)
        # asymmetric_M = freq_matrix_dic[setting_code]


    for i in range(1):
        ax.append(fig.add_subplot(1,1,i+1))

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'density':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-4, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-4, 'vmax': 1e-1}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['W'] = {'vmin': 1e-4, 'vmax': 1e-1}

        elif density_or_frequency == 'frequency':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}

        im = ax[0].imshow(asymmetric_M.T, origin = 'lower', interpolation = 'nearest', cmap = cmap, norm = LogNorm(vmin = vbounds[setting_code]['vmin'], vmax = vbounds[setting_code]['vmax']))

    else:

        im = ax[0].imshow(asymmetric_M.T, origin = 'lower', interpolation = 'nearest', cmap = cmap)

    implot.append(im)


    for i in range(1):
        divider = make_axes_locatable(ax[i])
        cax.append( divider.new_horizontal(size = "4%", pad = 0.15) )

        fig.add_axes(cax[i])
        cbar.append(fig.colorbar(implot[i], cax = cax[i]))
        cbar[i].ax.tick_params(axis = 'y', labelsize = 20)
        if density_or_frequency == 'frequency':
            cbar[i].ax.set_ylabel('Frequency of Contacts', fontsize = 18)
        else:
            cbar[i].ax.set_ylabel('Density of Contacts', fontsize = 18)
        ax[i].tick_params(labelsize = 18)
        ax[i].set_xlabel('Age', fontsize = 22)
        ax[i].set_ylabel('Age of Contacts',fontsize = 22)
        ax[i].set_title(titles[setting_code] + ' Contact Patterns', fontsize = 26)

        if aggregate_flag:
            tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels,fontsize = 16)
            ax[i].set_xticklabels(tick_labels,fontsize = 16, rotation = 50)
            ax[i].set_yticks(np.arange(len(tick_labels)))
            ax[i].set_yticklabels(tick_labels, fontsize = 16)


    return fig

