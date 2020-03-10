import os

__all__ = ['datadir' ,'set_datadir']

username = os.path.split(os.path.expanduser('~'))[-1]
datadirdict = {
    'dmistry': os.path.join('/home','dmistry','Dropbox (IDM)','seattle_network'),
    'cliffk': '/home/cliffk/idm/covid-19/data/seattle_network',
}

if username in datadirdict.keys():
    datadir = datadirdict[username]
else:
    print(f'Note: your username, "{username}", is not configured. Please use synthpops.set_datadir() to configure.')


def set_datadir(folder):
    ''' Set the folder to something user-specific '''
    datadir = folder
    print(f'Done; data directory set to {folder}')
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
