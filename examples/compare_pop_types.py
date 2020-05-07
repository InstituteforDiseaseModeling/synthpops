import sciris as sc
import covasim as cv
import synthpops as sp
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
import cmocean
import os


# def calculate_age_mixing_matrix


if __name__ == '__main__':
    

    pars = sc.objdict(
        pop_size  = 10000,
        n_days    = 1,
        # rand_seed = 1,
        # pop_type  = 'hybrid',
        pop_type  = 'synthpops',
    )

    sim             = cv.Sim(pars=pars)

    #### INTERVENTIONS GO HERE ####

    # sim.update_pars()
    # sim.init_interventions()

    sim.run(verbose = False)
    people          = sim.people

    ages      = sim.people.age
    ages = np.round(ages, 1)
    ages = ages.astype(int)

    max_age   = max(ages)
    age_count = Counter(ages)
    age_range = np.arange(max_age+1)

    # layer = 'w'
    # layer  = 's'

    plot_matrix     = True
    # plot_matrix     = False

    # plot_n_contacts = True
    # plot_n_contacts = False

    # cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
    cmap = mplt.cm.get_cmap(cmocean.cm.matter_r)

    if plot_matrix:

        n_contacts_count = np.zeros(max_age+1)
        symmetric_matrix = np.zeros((max_age+1, max_age+1))

        for p in range(len(sim.people)):

            for layer in ['h', 's', 'w']:
                contacts = sim.people.contacts[layer]['p2'][sim.people.contacts[layer]['p1'] == p]
                n_contacts = (sim.people.contacts[layer]['p1'] == p).sum()
                contact_ages = ages[contacts]

                for ca in contact_ages:
                    symmetric_matrix[ages[p]][ca] += 1
                n_contacts_count[ages[p]] += n_contacts

        age_brackets = sp.get_census_age_brackets(sp.datadir,state_location='Washington',country_location='usa')
        age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

        aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)
        aggregate_matrix = symmetric_matrix.copy()
        aggregate_matrix = sp.get_aggregate_matrix(aggregate_matrix, age_by_brackets_dic)

        asymmetric_matrix = sp.get_asymmetric_matrix(aggregate_matrix, aggregate_age_count)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        im = ax.imshow(asymmetric_matrix.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=1e-1, vmax=1e1))
        # im = ax.imshow(asymmetric_matrix.T, origin='lower', interpolation='nearest', cmap=cmap, )

        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="4%", pad=0.15)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(axis='y', labelsize=20)
        tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=16)
        ax.set_xticklabels(tick_labels, fontsize=16, rotation=50)
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=16)

        # fig.savefig('data/'+pars['pop_type'] + '_' + str(pars['pop_size']) + '_' + layer + '_contact_matrix.pdf', format='pdf')
        fig_name = os.path.join("..", "data", f"{pars['pop_type']}_{pars['pop_size']}_total_contact_matrix_day_{pars['n_days']}.png")
        fig.savefig(fig_name)
        plt.show()

