"""
Test advanced regression method
"""
import numpy as np
import sciris as sc
import synthpops as sp
import covasim as cv
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns


pars = sc.objdict(
    n                       = 5e3,
    rand_seed               = 123,
    smooth_ages             = 1,

    household_method = 'fixed_ages',

    with_facilities         = 1,
    with_non_teaching_staff = 1,
    with_school_types       = 1,

    school_mixing_type      = {'pk': 'random',
                          'es': 'random',
                          'ms': 'random',
                          'hs': 'random',
                          'uv': 'random'
                          },

)


def test_count_layer_degree():

    pop = sp.Pop(**pars)

    layers = 'H'
    # layers = ['H', 'LTCF']
    ages = None
    uids = None
    # uids = [0, 1]
    # ages = [95, 89, 90, 91]
    degree_df = sp.count_layer_degree(pop, layers, ages, uids)
    # sc.tic()
    # stats = sp.compute_layer_degree_statistics(pop, degree_df=degree_df)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    # cmap = mplt.cm.get_cmap('mako_r')

    g = sns.jointplot(x="age", y="degree", data=degree_df, cmap=cmap, alpha=0.75,
                      xlim=[0, 101], kind='kde', shade=True, thresh=0.01,
                      color=cmap(0.9), ylim=[0, 6],
                      space=0).plot_marginals(sns.kdeplot, color=cmap(0.75),
                                              shade=True, alpha=.5, legend=False)

    test_pars = sc.dcp(pars)
    test_pars['rand_seed'] = 0

    pop2 = sp.Pop(**test_pars)
    # degree_df2 = sp.count_layer_degree(pop2, layers, ages, uids)
    # cmap2 = sns.cubehelix_palette(rot=0.1, light=1, as_cmap=True)
    # g.plot_joint(sns.kdeplot, data=degree_df2, cmap=cmap2, alpha=0.75, xlim=[0, 101],
    #             kind='kde', shade=True, thresh=0.01, color=cmap2(0.9), ylim=[0, 6], 
    #             space=0).plot_marginals(sns.kdeplot, color=cmap2(0.75), shade=True, alpha=0.5, legend=False)


    plt.show()

    return pop




if __name__ == '__main__':

    test_count_layer_degree()
