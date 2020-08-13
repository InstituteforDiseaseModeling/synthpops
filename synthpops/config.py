'''
This module sets the location of the data folder and other global settings.

To change the level of log messages displayed, use e.g.

    sp.logger.setLevel('CRITICAL')
'''

#%% Housekeeping

import os
import sciris as sc
import logging
import sys

__all__ = ['logger', 'datadir', 'localdatadir', 'set_datadir', 'set_nbrackets', 'validate']

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
nbrackets = [16, 20][1] # Choose how many age bins to use -- 20 is only partially supported
matrix_size = 16 # The dimensions of the mixing matrices -- currently only 16 is available


#%% Logger -- adapted from Atomica

# Set the default logging level
default_log_level = ['DEBUG', 'INFO', 'WARNING', 'CRITICAL'][0]

logger = logging.getLogger('synthpops')

if not logger.hasHandlers():
    # Only add handlers if they don't already exist in the module-level logger
    # This means that it's possible for the user to completely customize *a* logger called 'atomica'
    # prior to importing Atomica, and the user's custom logger won't be overwritten as long as it has
    # at least one handler already added. The use case was originally to suppress messages on import, but since
    # importing is silent now, it doesn't matter so much.
    debug_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
    info_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
    warning_handler = logging.StreamHandler(sys.stderr)  # warning_handler will send all messages at or above WARNING to STDERR

    # Handle levels
    debug_handler.setLevel(0)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
    info_handler.setLevel(logging.INFO)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
    warning_handler.setLevel(logging.WARNING)
    debug_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.INFO})())  # Display anything INFO or higher
    info_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.WARNING})())  # Don't display WARNING or higher

    # Set formatting and log level
    formatter = logging.Formatter('%(levelname)s %(asctime)s.%(msecs)d {%(filename)s:%(lineno)d} - %(message)s', datefmt='%H:%M:%S')
    for handler in [debug_handler, info_handler, warning_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(default_log_level)  # Set the overall log level


def checkmem(unit='mb', fmt='0.2f', start=0, to_string=True):
    ''' For use with logger, check current memory usage '''
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mapping = {'b':1, 'kb':1e3, 'mb':1e6, 'gb':1e9}
    try:
        factor = mapping[unit.lower()]
    except KeyError:
        raise sc.KeyNotFoundError(f'Unit {unit} not found')
    mem_use = process.memory_info().rss/factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else:
        output = mem_use
    return output


#%% Functions

logger.debug(f'SynthPops location: {thisdir}')
logger.debug(f'Data folder: {datadir}')

def set_datadir(folder):
    '''Set the data folder to the user-specified location -- note, mostly deprecated.'''
    global datadir
    datadir = folder
    logger.info(f'Done: data directory set to {folder}.')
    return datadir

def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16 or 20.'''
    global nbrackets
    nbrackets = n
    if nbrackets not in [16, 20]:
        print(f'Note: current supported bracket choices are 16 or 20, use {nbrackets} at your own risk.')
    logger.info(f'Done: number of brackets is set to {n}.')
    return nbrackets


def validate(verbose=True):
    ''' Check that the data folder can be found. '''
    if os.path.isdir(datadir):
        if verbose:
            logger.debug(f'The data folder {datadir} was found.')
    else:
        if datadir is None:
            raise FileNotFoundError(f'The datadir has not been set; use synthpops.set_datadir() and try again.')
        else:
            raise FileNotFoundError(f'The folder "{datadir}" does not exist, as far as I can tell.')
