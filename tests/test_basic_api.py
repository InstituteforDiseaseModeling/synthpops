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
outfile = 'basic_api.pop'

pars = settings.get_full_feature_pars()


def test_basic_api():
    ''' Basic SynthPops test '''
    sp.logger.info('Testing basic API')

    pop = sp.make_population(**pars)
    age_distr = sp.read_age_bracket_distr(sp.default_config.datadir, country_location='usa', state_location='Washington', location='seattle_metro')
    assert len(age_distr) == 20, f'Check failed, len(age_distr): {len(age_distr)}'  # will remove if this passes in github actions test
    if regenerate or not os.path.exists(outfile):
        print('Saving...')
        sc.saveobj(outfile, pop)
    else:
        print('Checking...')
        pop2 = sc.loadobj(outfile)
        print(len(pop), len(pop2))
        assert pop == pop2, 'Check failed'
        print('Check passed')
    return pop


if __name__ == '__main__':
    T = sc.tic()
    pop = test_basic_api()
    sc.toc(T)
    print('Done.')
