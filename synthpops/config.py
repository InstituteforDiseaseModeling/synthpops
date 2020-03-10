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
    print(f'Note: your username, "{username}", is not configured. Please use synthpop.set_datadir() to configure.')

def set_datadir(folder):
    datadir = folder
    print(f'Done; data directory set to {folder}')
    return

