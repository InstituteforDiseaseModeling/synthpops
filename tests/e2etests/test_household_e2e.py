import pathlib
import pytest
import sciris as sc
import synthpops as sp
import setup_e2e as e2e
from setup_e2e import create_sample_pop, get_fig_dir_by_module


global sample_pop
def test_househould_average_contact_by_age(do_show, do_save, create_sample_pop, get_fig_dir_by_module):
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    fig = create_sample_pop.plot_contacts(**plotting_kwargs)
    fig.savefig(pathlib.Path(get_fig_dir_by_module, "household_age_mixing.png"))


def test_age_distribution():
    #todo: require statistics methods
    pass

def test_household_distribution():
    # todo: require statistics methods
    pass

def test_household_head_age_distribution(do_show, do_save, create_sample_pop, get_fig_dir_by_module):
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    sp.plotting.plot_household_head_age_dist_by_family_size(create_sample_pop, **plotting_kwargs)

if __name__ == "__main__":
    pytest.main(['-v', __file__])