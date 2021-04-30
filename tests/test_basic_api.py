'''
Simple run of the main API.

To create the test file, set regenerate = True. To test if it matches the saved
version, set regenerate = False.
'''

import os
import sciris as sc
import synthpops as sp
import settings

regenerate = False
# regenerate = True
outfile = 'basic_api.pop'

pars = settings.get_full_feature_pars()


def test_basic_api():
    ''' Basic SynthPops test '''
    sp.logger.info('Testing basic API')

    # pop = sp.make_population(**pars)
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


if __name__ == '__main__':
    T = sc.tic()
    pop = test_basic_api()
    sc.toc(T)
    print('Done.')
