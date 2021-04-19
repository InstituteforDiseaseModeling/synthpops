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

    layer = 'S'
    ages = None
    uids = None
    degree_df = sp.count_layer_degree(pop, layer, ages, uids)

    stats = sp.compute_layer_degree_statistics(pop, degree_df=degree_df)
    return pop


def test_plot_degree_by_age_methods(layer='S', do_show=False, do_save=False):
    sp.logger.info("Testing the different plotting methods to show the degree distribution by age for a single population.")
    # age on x axis, degree distribution on y axis
    pop = sp.Pop(**pars)
    kwargs = sc.objdict(do_show=do_show, do_save=do_save)

    ages = None
    uids = None

    uids_included = None

    degree_df = sp.count_layer_degree(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included)

    # kde seaborn jointplot
    gkde = sp.plotting.plot_degree_by_age(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included, degree_df=degree_df, kind='kde', **kwargs)

    # hist seaborn jointplot
    ghist = sp.plotting.plot_degree_by_age(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included, degree_df=degree_df, kind='hist', **kwargs)

    # reg seaborn joint
    greg = sp.plotting.plot_degree_by_age(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included, degree_df=degree_df, kind='reg', **kwargs)

    # hex seaborn joint
    # extra features: can limit the ages or uids, then just don't include the degree_df and it will calculate a new one with those filters applied
    age = np.arange(3, 76)
    kwargs.xlim = [3, 75]
    ghex = sp.plotting.plot_degree_by_age(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included, kind='hex', **kwargs)

    ages = np.arange(15, 76)
    fig, axboxplot = sp.plotting.plot_degree_by_age_boxplot(pop, layer='W', ages=ages, **kwargs)

    return gkde, ghist, greg, ghex, axboxplot


def test_multiple_degree_histplots(layer='S', do_show=False, do_save=False):
    sp.logger.info("Testing a plotting dev tool to compare the degree distribution by age for multiple populations.")

    npops = 6
    pop_list = []

    for ni in range(npops):
        pars_i = sc.dcp(pars)
        pars_i['rand_seed'] = ni

        pop_list.append(sp.Pop(**pars_i))

    kwargs = sc.objdict()
    kwargs.figname = f"multiple_degree_distribution_by_age_layer_{layer}"
    kwargs.do_show = do_show
    kwargs.do_save = do_save
    kind = 'kde'
    fig, axes = sp.plotting.plot_multi_degree_by_age(pop_list, layer=layer, kind=kind, **kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not made.'
    print('Check passed. Multi pop degree distribution plot made - 2D kde style.')

    kind = 'hist'
    fig2, axes2 = sp.plotting.plot_multi_degree_by_age(pop_list, layer=layer, kind=kind, **kwargs)
    assert isinstance(fig2, mplt.figure.Figure), 'Check 2 failed. Figure not made.'
    print('Check passed. Multi pop degree distribution plot made - 2D hist style.')

    return fig, axes, fig2, axes2


if __name__ == '__main__':

    # test_count_layer_degree()

    # test_multiple_degree_histplots(do_show=True)

    gkde, ghist, greg, ghex, axboxplot = test_plot_degree_by_age_methods(do_show=True)
