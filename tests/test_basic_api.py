'''
Simple run of the main API.

To create the test file, set regenerate = True. To test if it matches the saved
version, set regenerate = False.
'''

import os
import sciris as sc
import synthpops as sp
import settings
import pytest
import logging

regenerate = False
# regenerate = True
outfile = 'basic_api.pop'

pars = settings.get_full_feature_pars()


def test_basic_api():
    ''' Basic SynthPops test '''
    sp.logger.info('Testing basic API')

    pop = sp.Pop(**pars)
    popdict = pop.popdict
    age_distr = sp.read_age_bracket_distr(sp.settings.datadir, country_location='usa', state_location='Washington', location='seattle_metro')
    assert len(age_distr) == 20, f'Check failed, len(age_distr): {len(age_distr)}'  # will remove if this passes in github actions test
    if regenerate or not os.path.exists(outfile):
        print('Saving...')
        sc.saveobj(outfile, popdict)
    else:
        print('Checking...')
        popdict2 = sc.loadobj(outfile)
        print(len(popdict), len(popdict2))
        assert popdict == popdict2, f'Check failed, {pop.summarize()}'
        print('Check passed')
    return pop


def test_pop_n():
    """Test when n is None."""
    sp.logger.info("Testing when n is None.")
    test_pars = sc.dcp(pars)
    test_pars['n'] = None
    pop = sp.Pop(**test_pars)
    assert pop.n == sp.defaults.default_pop_size, 'Check failed.'
    print('Check passed')


def test_small_pop_n(caplog):
    """Test for when n is too small to make a population that makes sense."""
    sp.logger.info("Testing when n is small.")
    test_pars = sc.dcp(pars)
    test_pars['n'] = sp.defaults.default_pop_size - 1
    with caplog.at_level(logging.WARNING):
        pop2 = sp.Pop(**test_pars)
        assert pop2.n == sp.defaults.default_pop_size - 1, 'Check failed.'
        assert len(caplog.text) != 0, 'Check failed. No warning message about small pop size n.'
        assert len(pop2.ltcfs) == pop2.n_ltcfs == 0, 'Check failed. LTCF should not be created by small pop size n.'
    print('Check passed')


if __name__ == '__main__':
    T = sc.tic()
    pop = test_basic_api()
    test_pop_n()

    sc.toc(T)
    print('Done.')
