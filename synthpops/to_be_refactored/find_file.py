import datetime
import os
#from pathlib import Path
#import ntpath
#import re
import numpy as np
import pandas as pd
import sciris as sc
from collections import Counter
import base as spb
import config as cfg
import base as spb
import data_distributions as spdata

if __name__ == "__main__":

    altdirs = cfg.datadir
    root_check = 'exists' if os.path.isdir(cfg.datadir)  else 'does not exists'
    print(f"root_dir  {cfg.datadir}  {root_check}")

    #postal code check
    country_location = 'usa'
    state_location = 'Washington'
    location = None

    postal_codes = spdata.get_state_postal_code(state_location, country_location)
    print(len(postal_codes))

    # test existing data
    #expected = os.path.join(cfg.datadir, "usa", "Washington", LongTermCare_Table_48_Part{1}_{1}_2015_2016')

    """
    test_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    print('test_root=', test_root)
    #print('test_datadir_raw =',os.path.join(test_root, 'tests', 'test_data', 'demographics', 'contact_matrices_152_countries'))
    test_datadir = os.path.join(test_root, 'tests', 'test_data', 'demographics', 'contact_matrices_152_countries')
    print("test_datadir = ", test_datadir)
    root_check = 'exists' if os.path.isdir(test_datadir)  else 'does not exists'
    print(f"test_datadir  {cfg.datadir}  {root_check}")

    cfg.set_datadir(test_datadir)
    if cfg.datadir != test_datadir:
        print(f'Set directory test failed! {cfg.datadir} not equal to {test_datadir}.')
    else:
        print(f'Set directory test passed!')


    location_dirs = cfg.FilePaths('Dakar', 'Dakar', 'Senegal', root_dir=cfg.datadir)
    print(location_dirs.basedirs)

    print()
    print()
    location_with_alt1 = cfg.FilePaths('Dakar', 'Dakar', 'Senegal', 'usa',' Washington', 'seattle_metro', root_dir=cfg.datadir)
    # should get only 3 since alt_rootdir is not define
    print('alt location without altdir {0}'.format('Passed' if len(location_with_alt1.basedirs) == 3 else 'Failed'))

    print()
    print()
    location_with_alt2 = cfg.FilePaths('Dakar', 'Dakar', 'Senegal', 'seattle_metro', 'Washington', 'usa',root_dir=cfg.datadir, alt_rootdir=altdirs)
    print('alt location with altdir {0}'.format('Passed' if len(location_with_alt2.basedirs) == 6 else 'Failed'))

    #print()
    #print()
    #location_dirs = cfg.FilePaths('Dakar', 'Dakar', 'Senegal', root_dir=cfg.datadir)
    #print(location_dirs.basedirs)

    #print()
    #print()
    #location_dirs.get_demographic_file('age_distributions')
    #print()
    #print()
    #print('get_files')
    #cfg.set_datadir(altdirs)
    #country_location = 'usa'
    #state_location = 'Washington'
    #files = os.path.join(cfg.datadir, country_location, state_location, 'age_distributions', state_location + f'_gender_fraction_by_age_bracket_{cfg.nbrackets}.dat')

    #file = location_dirs.get_demographic_file('age_distributions', prefix=prefix, suffix='.dat', filter = ['16','18', '20'])
    #location_usa = cfg.FilePaths( 'seattle_metro', 'Washington', 'usa',root_dir=cfg.datadir)
    #location_files = location_usa.get_demographic_file('age_distributions', '{location}_gender_fraction_by_age_bracket_{nbrackets}')
    #print('files= ',files)
    #print('location_files=',location_files)
    location_dirs = cfg.FilePaths('Dakar', 'Dakar', 'Senegal', root_dir=cfg.datadir)
    location_files = location_dirs.get_demographic_file('age_distributions', '{location}_age_bracket_distr_{nbrackets}')
    """
