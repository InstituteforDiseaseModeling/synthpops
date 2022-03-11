"""Test Dakar location works with basic synthpops methodology."""
import sciris as sc
import synthpops as sp
import settings


default_nbrackets = sp.settings.nbrackets

pars = sc.objdict(
    n                               = settings.pop_sizes.small,
    rand_seed                       = 0,
    max_contacts                    = None,
    location                        = 'Dakar',
    state_location                  = 'Dakar',
    country_location                = 'Senegal',
    use_default                     = False,
    )


def test_Dakar():
    """Test that a Dakar population can be created with the basic SynthPops API."""
    sp.logger.info("Test that a Dakar population can be created with the basic SynthPops API.")
    sp.set_nbrackets(18)  # Dakar age distributions available are up to 18 age brackets
    pop = sp.Pop(**pars)
    popdict = pop.to_dict()
    assert len(popdict) == pars.n, 'Check failed.'
    print('Check passed')

    sp.set_location_defaults('defaults')  # Reset default values after this test is complete.

    sp.logger.info("Test that the default country was reset.")
    assert sp.settings.country_location == 'usa', f'Check failed: default_country is {sp.settings.country_location}'
    print('2nd Check passed')

    return pop


"""
Note:

This method does not include socioeconomic conditions that are likely
associated with school enrollment.

"""

if __name__ == '__main__':
    T = sc.tic()
    pop = test_Dakar()
    sc.toc(T)
    print(f"Dakar, Senegal population of size {pars['n']} made.")

    print('Done.')