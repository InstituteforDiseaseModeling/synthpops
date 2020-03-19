'''
The point of this file is to set the location of the data folder.
'''

#%% Housekeeping

import os

__all__ = ['datadir' ,'set_datadir', 'validate']

# Declaring this here makes it globally available as synthpops.datadir
datadir = None

# Set user-specific configurations
userpath = os.path.expanduser('~')
username = os.path.split(userpath)[-1]
datadirdict = {

    # 'dmistry': '/home/dmistry/Dropbox (IDM)/seattle_network',
    'dmistry': os.path.join(userpath,'Dropbox (IDM)','synthpops'),
    'cliffk': os.path.join(userpath,'idm','Dropbox','synthpops'),
    'lgeorge': os.path.join(userpath,'Dropbox','synthpops'),
    'ccollins': os.path.join(userpath,'Dropbox','synthpops')
}

# Try to find the folder on load
if username in datadirdict.keys():
    datadir = datadirdict[username]
    print(f'Loading synthpops data from {datadir}.')
else:
    print(f'synthpops: your username "{username}" is not configured. Please use synthpops.set_datadir() to configure.')


#%% Functions

def set_datadir(folder):
    ''' Set the folder to something user-specific '''
    global datadir
    datadir = folder
    print(f'Done; data directory set to {folder}.')
    return


def validate(verbose=True):
    ''' Check that the data folder can be found '''
    if os.path.isdir(datadir):
        if verbose:
            print(f'The data folder {datadir} was found.')
    else:
        if datadir is None:
            raise FileNotFoundError(f'The datadir has not been set; use synthpops.set_datadir() and try again.')
        else:
            raise FileNotFoundError(f'The folder "{datadir}" does not exist, as far as I can tell.')
