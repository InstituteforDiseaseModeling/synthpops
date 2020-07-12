'''
This module sets the location of the data folder.
'''

#%% Housekeeping

import os
import sciris as sc

__all__ = ['datadir', 'localdatadir', 'set_datadir', 'set_nbrackets', 'validate']

# Declaring this here makes it globally available as synthpops.datadir
datadir = None
localdatadir = None
full_data_available = False # this is likely not necesary anymore

# Set the local data folder
thisdir = sc.thisdir(__file__)
print(thisdir)
localdatadir = os.path.join(thisdir, os.pardir, 'data')

# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir


# Number of census age brackets to use
nbrackets = [16, 20][1] # Choose how many age bins to use -- 20 is only partially supported
matrix_size = 16 # The dimensions of the mixing matrices -- currently only 16 is available


#%% Functions

def set_datadir(folder):
    '''Set the data folder to the user-specified location -- note, mostly deprecated.'''
    global datadir
    datadir = folder
    print(f'Done: data directory set to {folder}.')
    return datadir

def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16 or 20.'''
    global nbrackets
    nbrackets = n
    if nbrackets not in [16, 20]:
        print(f'Note: current supported bracket choices are 16 or 20, use {nbrackets} at your own risk.')
    print(f'Done: number of brackets is set to {n}.')
    return nbrackets


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
