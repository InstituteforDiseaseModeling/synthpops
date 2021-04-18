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
    n                       = 50e3,
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

    # layers = 'H'
    # layers = ['H', 'LTCF']
    # layers = 'S'
    layer = 'S'
    ages = None
    uids = None
    # uids = [0, 1]
    # ages = [95, 89, 90, 91]
    # ages = np.arange(5, 25)
    degree_df = sp.count_layer_degree(pop, layer, ages, uids)
    print(degree_df.loc[degree_df['age'] == 20])

    # sc.tic()
    # stats = sp.compute_layer_degree_statistics(pop, degree_df=degree_df)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    # cmap = mplt.cm.get_cmap('mako_r')

    # g = sns.jointplot(x="age", y="degree", data=degree_df, cmap=cmap, alpha=0.75,
    #                   xlim=[0, 101], kind='kde', shade=True, thresh=0.01,
    #                   color=cmap(0.9), ylim=[0, 6],
    #                   space=0).plot_marginals(sns.kdeplot, color=cmap(0.75),
    #                                           shade=True, alpha=.5, legend=False)

    kwargs = sc.objdict()
    kwargs.xlim = [4, 30]
    kwargs.figname='hex_degree'
    kwargs.do_save=1

    # kind = 'kde'
    # kind = 'hist'
    # kind = 'reg'
    kind = 'hex'

    g = sp.plot_degree_by_age(pop, layers=layer, degree_df=degree_df, kind=kind, **kwargs)


    # fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
    # fig.subplots_adjust(left=0.06, right=0.97, bottom=0.15)

    # pars2 = sc.dcp(pars)
    # pars2['rand_seed'] = 0
    # alpha = 0.95
    # thresh = 0.001

    # pop2 = sp.Pop(**pars2)
    # degree_df2 = sp.count_layer_degree(pop2, layer, ages, uids)
    # cmap2 = sns.cubehelix_palette(rot=0.3, light=1, as_cmap=True)

    # pars3 = sc.dcp(pars)
    # pars3['rand_seed'] = 1

    # pop3 = sp.Pop(**pars3)
    # degree_df3 = sp.count_layer_degree(pop3, layer, ages, uids)
    # cmap3 = sns.cubehelix_palette(rot=0.2, light=1, as_cmap=True)

    # sns.kdeplot(x=degree_df['age'], y=degree_df['degree'], cmap=cmap, shade=True, ax=axes[0],
    #             alpha=alpha, thresh=thresh, cbar=True)
    # sns.kdeplot(x=degree_df2['age'], y=degree_df2['degree'], cmap=cmap2, shade=True, ax=axes[1],
    #             alpha=alpha, thresh=thresh, cbar=True)
    # sns.kdeplot(x=degree_df3['age'], y=degree_df3['degree'], cmap=cmap3, shade=True, ax=axes[2],
    #             alpha=alpha, thresh=thresh, cbar=True)
    # # g.plot_joint(sns.kdeplot, data=degree_df2, cmap=cmap2, alpha=0.75, xlim=[0, 101],
    # #             kind='kde', shade=True, thresh=0.01, color=cmap2(0.9), ylim=[0, 6], 
    # #             space=0).plot_marginals(sns.kdeplot, color=cmap2(0.75), shade=True, alpha=0.5, legend=False)

    # for ax in axes:
    #     ax.set_xlim(0, 101)
    #     ax.set_ylim(0, 6)

    plt.show()

    return pop








if __name__ == '__main__':

    test_count_layer_degree()
