'''
This module sets the location of the data folder.
'''

#%% Housekeeping

import os
import sciris as sc

__all__ = ['datadir', 'localdatadir', 'set_datadir', 'validate']

# Declaring this here makes it globally available as synthpops.datadir
datadir = None
localdatadir = None
full_data_available = False # this is likely not necesary anymore

# Set the local data folder
thisdir = sc.thisdir(__file__)
localdatadir = os.path.join(thisdir, os.pardir, 'data')

# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir


# Number of census age brackets to use
nbrackets = [16, 20][0] # 20 is only partially supported


#%% Functions

def set_datadir(folder):
    '''Set the data folder to the user-specified location.'''
    global datadir
    datadir = folder
    print(f'Done: data directory set to {folder}.')
    return datadir


def validate(verbose=True):
    ''' Check that the data folder can be found. '''
    if os.path.isdir(datadir):
        if verbose:
            print(f'The data folder {datadir} was found.')
    else:
        if datadir is None:
            raise FileNotFoundError(f'The datadir has not been set; use synthpops.set_datadir() and try again.')
        else:
            raise FileNotFoundError(f'The folder "{datadir}" does not exist, as far as I can tell.')
