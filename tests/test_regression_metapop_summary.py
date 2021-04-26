"""
Test advanced regression method
"""
import numpy as np
import sciris as sc
import synthpops as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import settings
import pytest

mplt_org_backend = mplt.rcParamsDefault['backend']  # interactive backend for user
mplt.use('Agg')


pars = sc.objdict(
    n                       = settings.pop_sizes.medium_large,
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


@pytest.fixture(scope="module")
def create_pop():
    return sp.Pop(**pars)


def test_pop_summarize(create_pop):
    sp.logger.info("Test that pop.summarize() works.")
    pop = create_pop
    summary_msg = pop.summarize(return_msg=True)
    assert f"The number of people is {pars.n:.0f}" in summary_msg


def test_count_layer_degree(create_pop):
    sp.logger.info("Testing degree_df and quantile stats calculating method for degree distribution by age.")
    pop = create_pop

    layer = 'S'
    ages = None
    uids = None
    degree_df = sp.count_layer_degree(pop, layer, ages, uids)
    assert list(degree_df.columns.values) == ['uid', 'age', 'degree', 'contact_ages'], 'Check failed.'
    print('Check passed.')

    desc = sp.compute_layer_degree_description(pop, degree_df=degree_df)
    cols = list(desc.columns.values)
    expected_cols = ['count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max']
    for exc in expected_cols:
        assert exc in cols, f'Check failed. {exc} not found in description columns.'
    print('Check passed. Found all expected columns.')

    return pop


def test_plot_degree_by_age_methods(create_pop, layer='S', do_show=False, do_save=False):
    sp.logger.info("Testing the different plotting methods to show the degree distribution by age for a single population.")
    # age on x axis, degree distribution on y axis
    pop = create_pop
    kwargs = sc.objdict(do_show=do_show, do_save=do_save)

    ages = None
    uids = None

    uids_included = None

    degree_df = sp.count_layer_degree(pop, layer=layer, ages=ages, uids=uids, uids_included=uids_included)

    if kwargs.do_show:
        plt.switch_backend(mplt_org_backend)

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

    if kwargs['do_show']:
        plt.switch_backend(mplt_org_backend)

    kind = 'kde'
    fig, axes = sp.plotting.plot_multi_degree_by_age(pop_list, layer=layer, kind=kind, **kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not made.'
    print('Check passed. Multi pop degree distribution plot made - 2D kde style.')

    kind = 'hist'
    fig2, axes2 = sp.plotting.plot_multi_degree_by_age(pop_list, layer=layer, kind=kind, **kwargs)
    assert isinstance(fig2, mplt.figure.Figure), 'Check 2 failed. Figure not made.'
    print('Check passed. Multi pop degree distribution plot made - 2D hist style.')

    return fig, axes, fig2, axes2


def test_plot_degree_by_age_stats(create_pop, do_show=False, do_save=False):

    sp.logger.info("Testing plots of the statistics on the degree distribution by age summaries.")

    pop = create_pop
    kwargs = dict(do_show=do_show, do_save=do_save)
    if kwargs['do_show']:
        plt.switch_backend(mplt_org_backend)
    fig, ax = sp.plotting.plot_degree_by_age_stats(pop, **kwargs)
    plt.switch_backend('agg')

    return fig, ax


if __name__ == '__main__':

    # test_pop_summarize(create_pop)
    # test_count_layer_degree(create_pop)
    test_multiple_degree_histplots(do_show=1)
    # gkde, ghist, greg, ghexs, axboxplot = test_plot_degree_by_age_methods(create_pop, do_show=1)
    # fig, ax = test_plot_degree_by_age_stats(do_show=1)
