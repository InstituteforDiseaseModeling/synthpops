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

# Set user-specific configurations
userpath = os.path.expanduser('~')
username = os.path.split(userpath)[-1]
datadirdict = {
    # 'dmistry':  os.path.join(userpath,'Dropbox (IDM)','synthpops'),
    # 'cliffk':   os.path.join(userpath,'idm','Dropbox','synthpops'),
    # 'lgeorge':  os.path.join(userpath,'Dropbox','synthpops'),
    # 'ccollins': os.path.join(userpath,'Dropbox','synthpops')
}

# Try to find the folder on load
if username in datadirdict.keys():
    full_data_available = True
    datadir = datadirdict[username]
    if not os.path.isdir(datadir):
        errormsg = f'Your username "{username}" was found, but the folder {datadir} does not exist. Please fix synthpops/config.py and try again.'
        raise FileNotFoundError(errormsg)

# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir


#%% Functions

def set_datadir(folder):
    ''' Set the data folder to the user-specified location.'''
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
