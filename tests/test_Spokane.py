"""Test Spokane location works."""
import sciris as sc
import synthpops as sp
import matplotlib.pyplot as plt
import settings


pars = sc.objdict(
    n                               = settings.pop_sizes.medium,
    rand_seed                       = 123,
    location                        = 'Spokane_County',
    # location = 'seattle_metro',
    state_location                  = 'Washington',
    country_location                = 'usa',
    use_default                     = 1,  # search for seattle metro data for ltcfs

    household_method                = 'fixed_ages',

    with_facilities                 = 1,
    with_school_types               = 1,
    school_mixing_type              = 'age_and_class_clustered',
    )


def test_Spokane():
    """Test that a Dakar population can be created with the basic SynthPops API."""
    sp.logger.info("Test that a Spokane population can be created with the basic SynthPops API.")

    pop = sp.Pop(**pars)
    loc_pars = pop.loc_pars
    age_dist = sp.read_age_bracket_distr(**loc_pars)
    assert len(age_dist) == 20, f'Check failed, len(age_dist): {len(age_dist)}'  # will remove if this passes in github actions test

    sp.set_location_defaults('defaults')  # Reset default values after this test is complete.
    return pop


"""
Notes:

data missing:

ltcf resident sizes -- copied facility sizes from Seattle, King County to Spokane County.

"""

if __name__ == '__main__':
    T = sc.tic()
    pop = test_Spokane()
    sc.toc(T)
    print(f"Spokane County population of size {pars.n} made.")
