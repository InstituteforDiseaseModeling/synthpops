'''
This module sets the location of the data folder and other global settings.

To change the level of log messages displayed, use e.g.

    sp.logger.setLevel('CRITICAL')
'''

# %% Housekeeping

import os
import sys
import psutil
import sciris as sc
import logging
from . import version as spv

__all__ = ['logger', 'checkmem', 'datadir', 'localdatadir', 'rel_path', 'alt_rel_path', 'set_nbrackets',
           'validate', 'set_location_defaults', 'default_country', 'default_state',
           'default_location', 'default_sheet_name', 'alt_location', 'default_household_size_1_included',
           'get_config_data', 'version_info', 'max_age']


# Declaring this here makes it globally available as synthpops.datadir
datadir = None
alt_datadir = None
localdatadir = None
rel_path = []
alt_rel_path = []
full_data_available = False  # this is likely not necessary anymore

# Set the local data folder
thisdir = os.path.dirname(os.path.abspath(__file__))

localdatadir = os.path.abspath(os.path.join(thisdir, os.pardir, 'data'))


# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir

# Number of census age brackets to use
max_age = 101
valid_nbracket_ranges = [16, 18, 20]  # Choose how many age bins to use -- 20 is only partially supported
nbrackets = 20
matrix_size = 16  # The dimensions of the mixing matrices -- currently only 16 is available
default_country = None
default_state = None
default_location = None
default_sheet_name = None
alt_location = None
default_household_size_1_included = False

# %% Logger

# Set the default logging level
default_log_level = ['DEBUG', 'INFO', 'WARNING', 'CRITICAL'][1]

logger = logging.getLogger('synthpops')

if not logger.hasHandlers(): # pragma: no cover
    # Only add handlers if they don't already exist in the module-level logger
    # This means that it's possible for the user to completely customize *a* logger called 'synthpops'
    # prior to importing SynthPops, and the user's custom logger won't be overwritten as long as it has
    # at least one handler already added.
    debug_handler   = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
    info_handler    = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
    warning_handler = logging.StreamHandler(sys.stderr)  # warning_handler will send all messages at or above WARNING to STDERR

    # Handle levels
    debug_handler.setLevel(0)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
    info_handler.setLevel(logging.INFO)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
    warning_handler.setLevel(logging.WARNING)
    debug_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.INFO})())  # Display anything INFO or higher
    info_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.WARNING})())  # Don't display WARNING or higher

    # Set formatting and log level
    formatter = logging.Formatter('%(levelname)s %(asctime)s.%(msecs)d %(filename)s:%(lineno)d → %(message)s', datefmt='%H:%M:%S')
    debug_handler.setFormatter(formatter)
    for handler in [debug_handler, info_handler, warning_handler]:
        logger.addHandler(handler)
    logger.setLevel(default_log_level)  # Set the overall log level


def checkmem(unit='mb', fmt='0.2f', start=0, to_string=True):
    ''' For use with logger, check current memory usage '''
    process = psutil.Process(os.getpid())
    mapping = {'b': 1, 'kb': 1e3, 'mb': 1e6, 'gb': 1e9}
    try:
        factor = mapping[unit.lower()]
    except KeyError: # pragma: no cover
        raise sc.KeyNotFoundError(f'Unit {unit} not found')
    mem_use = process.memory_info().rss / factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else:
        output = mem_use
    return output


def get_config_data():
    data = {
        'valid_nbrackets':
            [16, 18, 20],
        'Senegal': {
             'location': 'Dakar',
             'province': 'Dakar',
             'country': 'Senegal',
             'sheet_name': 'Senegal',
             'nbrackets': 18,
             'household_size_1': True
              },
        'defaults': {
             'location': 'seattle_metro',
             'province': 'Washington',
             'country': 'usa',
             'sheet_name': 'United States of America',
             'nbrackets': 20
              },
        'usa': {
             'location': 'seattle_metro',
             'province': 'Washington',
             'country': 'usa',
             'sheet_name': 'United States of America',
             'nbrackets': 20
              }
         }
    return data


# %% Functions
def version_info():
    print(f'Loading SynthPops v{spv.__version__} ({spv.__versiondate__}) from {thisdir}')
    print(f'Data folder: {datadir}')
    print(f'Git information:')
    sc.pp(sc.gitinfo(__file__))
    return

def set_metadata(obj):
    ''' Set standard metadata for an object '''
    obj.version = spv.__version__
    obj.created = sc.now()
    obj.git_info = sc.gitinfo(__file__)
    return


def set_location_defaults(country=None):
    global config_file
    global default_country
    global default_state
    global default_location
    global default_sheet_name
    global nbrackets
    global default_household_size_1_included

    # read the confiuration file
    country_location = country if country is not None else 'defaults'
    data = get_config_data()

    if country_location in data.keys():
        loc = data[country_location]
        default_location = loc['location']
        default_state = loc['province']
        default_country = loc['country']
        default_sheet_name = loc['sheet_name']
        nbrackets = 20 if loc['nbrackets'] is None else loc['nbrackets']
        default_household_size_1_included = False if 'household_size_1' not in loc.keys() else loc['household_size_1']


set_location_defaults()


def set_datadir(root_dir, relative_path=None):
    '''Set the data folder and relative path to the user-specified
        location.
        On startup, the datadir and rel_path are set to the conventions
        used to store data. datadir is the root directory to the data, and
        rel_path is a list of sub directories to the data -->
        to change the location of the data the user is able to supply a new root_dir and new relative path. If the user uses a similar directory path model that we use
        e.g. root_dir/demographics/contact... the user can change datadir without changing relative path, by passing in relative_path = None (default)
        -- note, mostly deprecated.'''
    ''' user specifies complete path'''
    global datadir
    global rel_path
    datadir = root_dir
    if relative_path is not None:
        rel_path = relative_path
    logger.info(f'Done: data directory set to {root_dir}.')
    logger.info(f'Relative Path set to  {rel_path}.')
    return datadir


def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16 or 20.'''
    global nbrackets
    logger.info(f"set_nbrackets n = {n}")
    nbrackets = n
    if nbrackets not in valid_nbracket_ranges:
        logger.warningnt(f'Note: current supported bracket choices are {valid_nbracket_ranges}, use {nbrackets} at your own risk.')
    logger.info(f'Done: number of brackets is set to {n}.')
    return nbrackets


def validate():
    ''' Check that the data folder can be found. '''
    if os.path.isdir(datadir):
        logger.info(f"The data folder {datadir} was found.")

    else:
        if datadir is None:
            raise FileNotFoundError(f'The datadir has not been set; use synthpops.set_datadir() and try again.')
        else:
            raise FileNotFoundError(f'The folder "{datadir}" does not exist, as far as I can tell.')
