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
from . import defaults as spd


__all__ = ['logger',
           'checkmem',
           'set_nbrackets',
           'set_datadir',
           'validate_datadir',
           'set_location_defaults',
           'version_info',
           ]


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


# %% Functions
def version_info():
    print(f'Loading SynthPops v{spv.__version__} ({spv.__versiondate__}) from {spd.settings.thisdir}')
    print(f'Data folder: {spd.settings.datadir}')
    try:
        gitinfo = sc.gitinfo(__file__)
        print(f'Git information:')
        sc.pp(gitinfo)
    except:
        pass # Don't worry if git info isn't available
    return


def set_metadata(obj):
    ''' Set standard metadata for an object '''
    obj.version = spv.__version__
    obj.created = sc.now()
    obj.git_info = sc.gitinfo(__file__, verbose=False)
    return


def set_location_defaults(country_location=None):

    # read the confiuration file
    data = spd.default_data

    if country_location in data.keys():
        loc = data[country_location]

        spd.reset_settings(loc)
        # default_household_size_1_included = False if 'household_size_1' not in loc.keys() else loc['household_size_1']
        # spd.reset_settings_by_key('household_size_1_included', default_household_size_1_included)

    elif country_location is None:
        logger.debug(f"Setting default location information with {spd.default_data['defaults']}.")
        # logger.warning(f"Setting default location information with {spd.default_data['defaults']}.")  # we may want to set as a warning instead
        loc = data['defaults']
        spd.reset_settings(loc)
    else:
        logger.warning(f"synthpops has no defaults for {country_location}. You can use sp.reset_settings() to set the default location information for the keys: {spd.settings.keys()}")


# initially set defaults for the usa
set_location_defaults()


def set_datadir(root_dir, relative_path=None):
    '''
    Set the data folder and relative path to the user-specified location.

    On startup, the datadir and rel_path are set to the conventions used to
    store data. datadir is the root directory to the data, and relative_path is a
    list of sub directories to the data --> to change the location of the data
    the user is able to supply a new root_dir and new relative path. If the user
    uses a similar directory path model that we use e.g.
    root_dir/demographics/contact... the user can change datadir without
    changing relative path, by passing in relative_path = None (default) --
    note, mostly deprecated but still functional if needed.

    Args:
        root_dir (str)      : new root directory for the data folder to point to
        relative_path (str) : new relative path to the root_dir

    Returns:
        str: path to the updated settings.datadir
    '''
    datadir = root_dir
    if relative_path is not None:
        spd.reset_settings_by_key('relative_path', relative_path)

    logger.info(f'Done: data directory set to {root_dir}.')
    logger.info(f'Relative Path set to  {spd.settings.relative_path}.')

    spd.reset_settings_by_key('datadir', datadir)

    return spd.settings.datadir


def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16, 18 or 20.'''
    logger.info(f"set_nbrackets n = {n}")
    spd.reset_settings_by_key('nbrackets', n)

    if spd.settings.nbrackets not in spd.settings.valid_nbracket_ranges:
        logger.warning(f'Note: current supported bracket choices are {spd.settings.valid_nbracket_ranges}, use {spd.settings.nbrackets} at your own risk.')
    logger.info(f'Done: number of brackets is set to {n}.')

    return spd.settings.nbrackets


def validate_datadir(verbose=True):
    ''' Check that the data folder can be found. '''
    if os.path.isdir(spd.settings.datadir):
        logger.info(f"The data folder {spd.settings.datadir} was found.")
    else:
        if spd.settings.datadir is None:
            raise FileNotFoundError(f'The datadir has not been set; use synthpops.set_datadir() and try again.')
        else:
            raise FileNotFoundError(f'The folder "{spd.settings.datadir}" does not exist.')
