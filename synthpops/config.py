'''
This module sets the location of the data folder and other global settings.

To change the level of log messages displayed, use e.g.

    sp.logger.setLevel('CRITICAL')
'''

#%% Housekeeping

import os
import sys
import psutil
import sciris as sc
import logging
import yaml
from . import version as spv

__all__ = ['logger', 'checkmem', 'datadir', 'localdatadir', 'rel_path', 'alt_rel_path', 'set_nbrackets',
           'validate', 'set_location_defaults', 'default_country', 'default_state',
           'default_location', 'default_sheet_name', 'alt_location', 'default_household_size_1_included',
           'get_config_data', 'version_info']



# Declaring this here makes it globally available as synthpops.datadir
datadir = None
alt_datadir = None
localdatadir = None
rel_path = ['demographics', 'contact_matrices_152_countries']
alt_rel_path = ['demographics', 'contact_matrices_152_countries']
full_data_available = False # this is likely not necessary anymore

# Set the local data folder
thisdir = os.path.dirname(os.path.abspath(__file__))
#print(thisdir)
config_file = os.path.join(thisdir, 'config_info.yaml')
localdatadir = os.path.abspath(os.path.join(thisdir, os.pardir, 'data'))


# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir

# Number of census age brackets to use
valid_nbracket_ranges = [16, 18, 20] # Choose how many age bins to use -- 20 is only partially supported
nbrackets = 20
matrix_size = 16 # The dimensions of the mixing matrices -- currently only 16 is available
default_country = None
default_state = None
default_location = None
default_sheet_name = None
alt_location = None
default_household_size_1_included = False

#%% Logger

# Set the default logging level
default_log_level = ['DEBUG', 'INFO', 'WARNING', 'CRITICAL'][0]

logger = logging.getLogger('synthpops')

if not logger.hasHandlers():
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
    except KeyError:
        raise sc.KeyNotFoundError(f'Unit {unit} not found')
    mem_use = process.memory_info().rss / factor - start
    if to_string:
        output = f'{mem_use:{fmt}} {unit.upper()}'
    else:
        output = mem_use
    return output

def get_config_data(config_file):
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    return data

#%% Functions
def version_info():
    print(f'Loading SynthPops v{spv.__version__} ({spv.__versiondate__}) from {thisdir}')
    print(f'Data folder: {datadir}')
    print(f'Git information:')
    sc.pp(spv.__gitinfo__)
    return

def set_location_defaults(country=None):
    global config_file
    global default_country
    global default_state
    global default_location
    global default_sheet_name
    global nbrackets
    global default_household_size_1_included

    # read the yaml file
    country_location = country if country is not None else 'defaults'
    data = get_config_data(config_file)

    if country_location in data.keys():
        loc = data[country_location]
        default_location = loc['location']
        default_state = loc['province']
        default_country = loc['country']
        default_sheet_name = loc['sheet_name']
        nbrackets = 20 if loc['nbrackets'] is None else loc['nbrackets']
        default_household_size_1_included = False if 'household_size_1' not in loc.keys() else loc['household_size_1']


set_location_defaults()





def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16 or 20.'''
    global nbrackets
    logger.info(f"set_nbrackets n = {n}")
    nbrackets = n
    if nbrackets not in valid_nbracket_ranges:
        logger.warningnt(f'Note: current supported bracket choices are {valid_nbracket_ranges}, use {nbrackets} at your own risk.')
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



class FilePaths:
    """
    FilepPaths:
    FilePath builds a set of of possible path names for data file.
    The goal of FilePath is to build and ordered list of path names to
    search for the desired data target, and then return the first available target.

    On initialization the  primary location and file pattern
    are established. Once established, calls to get_file() will return will return the
    first available data file from the data tree. Optionally, the user can provide a alternate
    location.

    Note tis class assumes that the data is stored in a directory tree described above.



    Initialization:
        Required inputs:
            root_dir           -> root directory for one or all data sets
            location_info      -> country, province, location
            alt_location_info  -> alternate country, province, location
            pattern            -> the data file name pattern
    Methods:
        get_pattern_keys() - returns a list of keys defined in a PathPattern
        get_number_pattern_keys() - returns the number of keys defied in the PathPattern
        get_pattern_time_format(**kwargs) - returns the time format string for a set of kwargs
        get_path_pattern(*kwargs)
        get_file(data_type)
    Usage:
        pattern = '{prefix}_{type}.dat'
        root_dir = 'c:/documents'
        location_info = [country, province, location]
        path_pattern = PathPattern(root_dir, pattern, relitive_dir)
        keys = path_pattern.get_pattern_keys() # -> returns 'variable'
        nkeys = path_pattern.get_number_pattern_keys() #-> returns 1
        time_format path_pattern_time_format(variable='airtemp') #-> returns 'airtemp%Y%j.nc'
        file_pattern = path_pattern.get_path_pattern(variable='airtemp') # returns <root_dir>/<relitive_dir>/airtemp/%Y/airtemp%Y%j.nc
    """
    # note-change: add 'demographics', 'contact_matrices_152_countries' to root

    def __init__(self,  location=None, province=None, country=None,  alternate_location= None,  root_dir=None, alt_rootdir=None,  use_defaults=False):
        global datadir, alt_datadir, rel_path, alt_location
        base_dir = datadir
        if len(rel_path) > 0:
            base_dir= os.path.join(datadir, *rel_path)
        self.root_dir = base_dir if root_dir is None else root_dir

        self.alt_root_dir = alt_datadir if alt_rootdir is None else alt_rootdir
        self.country = country
        self.province = province
        self.location = location

        self.alt_country = None
        self.alt_province = None
        self.alt_location = None

        if self.alt_root_dir is None:
            self.alt_root_dir = self.root_dir
        self.basedirs = []

        self.add_base_location(location, province, country)

        if alternate_location is not None:
            logger.info(f"adding user call supplied alternate location ")
            self.add_alternate_location(location=alternate_location.location, province=alternate_location.state_location, country=alternate_location.country_location)
        elif alt_location is not None:
            logger.info(f"adding config alt location")
            self.add_alternate_location(location=alt_location.location, province=alt_location.state_location, country=alt_location.country_location)


    def add_base_location(self, location=None, province=None, country=None):

        levels = [location,province, country]
        if all(level is None for level in levels) :
            raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location, state_location, and country_location strings.")
        elif country is None:
            raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
        else:
            # build alternate dirs
            basedirs = self._add_dirs(self.root_dir, location, province, country)
            if len(basedirs) > 0:
                basedirs.reverse()
                self.basedirs.extend(basedirs)
        self.validate_dirs()


    def _add_dirs(self, root,location, province, country):
        pathdirs = []
        # build alternate dirs
        pathdirs.append((country, os.path.join(root, country)))
        if province is not None:
            pathdirs.append((province,os.path.join(root, country, province)))
            if location is not None:
                pathdirs.append((location,os.path.join(root, country, province, location)))
        return pathdirs

    def validate_dirs(self):
        # heck directories in base list and remove missing dirs from the list
        search_list = reversed(list(enumerate(self.basedirs)))
        for i,e in search_list:
            if not os.path.isdir(e[1]):
                logger.warning(f"Warning: Directory {e[1]} missing for location={i}, removing.")
                self.basedirs.pop(i)
        # make sure we have at least one directory, or through an error
        if len(self.basedirs) < 1:
             raise FileNotFoundError(f'The location data folders do not exist, as far as I can tell. Dirs tried = {search_list}')
             return False
        return True

    def get_demographic_file(self, location=None, filedata_type=None, prefix=None, suffix=None, filter_list=None, alt_prefix=None):
        """
        Search the base directories and return the first file found that matches the criteria
        """
        filedata_types = ['age_distributions', 'assisted_living', 'contact_networks', 'employment', 'enrollment', 'household_living_arrangements','household_size_distributions', 'schools', 'workplaces']
        if filedata_type is None:
            raise NotImplementedError(f"Missing filedata_type string.")
            return None

        if filedata_type not in filedata_types:
            raise NotImplementedError(f"Invalid filedata_type string {filedata_type}. filedata_type must be one of the following {filedata_types}")
            return None

        file = self._search_dirs(location, filedata_type, prefix, suffix, filter_list, alt_prefix)
        return file

    def get_data_file(self, location=None, filedata_type=None, prefix=None, suffix=None, filter_list=None, alt_prefix=None):
        """
        Search the base directories and return the first file found that matches the criteria
        """
        #filedata_type = None
        file = self._search_dirs(location, filedata_type, prefix, suffix, filter_list, alt_prefix)
        return file

    def _search_dirs(self, location, filedata_type, prefix, suffix, filter_list, alt_prefix):
        """
        Search the directories in self.basedirs for a file matches the conditions
        Location is the state_location, province, or city level if applicable
            (e.g. for usa, Washington state).
        Prefix is the start of the file name. Examples of prefix patterns are:
            '{location}_age_bracket_distr' or 'head_age_brackets'. If {location}
            appears in the prefix pattern, the country, province or location information
            will be substituted, depending on level.
        the suffix is the end of the file name. typically '.dat' for data file
        A list of matching file is returned.
        The number of files returned by prefix and suffix, can be reduced by using
        a filter pattern. the filter patter is a regex expression.
        """
        results = None
        for target in self.basedirs:
            files = None
            target_dir = target[1]
            target_location = target[0]
            if target_location == "seattle_metro":
                target_location = "Washington"
                target_dir = os.path.abspath(os.path.join(target[1], '..'))
            elif target_location == location:
                target_location = target[0]
            filedata_dir = target_dir
            if filedata_type is not None:
                filedata_dir = os.path.join(target_dir, filedata_type)

            # check if there is a directory
            if os.path.isdir(filedata_dir):
                if len(os.listdir(filedata_dir)) > 0:
                    files = self._list_files(target_location, filedata_dir, prefix, suffix, filter_list, alt_prefix)

                    if len(files) > 0:
                        results = os.path.join(filedata_dir, files[0])
                        break
                else:
                    logger.info(f'no data in directory {filedata_dir}, skipping')
        return results

    def _list_files(self, level, target_dir, prefix, suffix, filter_list, alt_prefix):
        global nbrackets
        prefix_pattern = prefix
        suffix_pattern = suffix
        alt_prefix_pattern = alt_prefix

        if level is not None and prefix_pattern is not None:
            prefix_pattern = prefix_pattern.format(location=level)

        # if level is not None and suffix is not None:
        #    suffix_pattern = suffix.format(location=level)
        files = []

        if prefix_pattern is not None and suffix_pattern is not None:
            files = [f for f in os.listdir(target_dir) if f.startswith(prefix_pattern) and f.endswith(suffix_pattern)]
        elif prefix_pattern is not None and suffix_pattern is None:
            files = [f for f in os.listdir(target_dir) if f.startswith(prefix_pattern)]
        elif prefix_pattern is None and suffix_pattern is not None:
            files = [f for f in os.listdir(target_dir) if f.endswith(suffix_pattern)]
        else:
            files = [f for f in os.listdir(target_dir)]

        # if if we have the file needed. if not check the alternate
        if len(files) == 0 and alt_prefix_pattern is not None:
            if alt_prefix_pattern is not None and suffix_pattern is not None:
                files = [f for f in os.listdir(target_dir) if f.startswith(alt_prefix_pattern) and f.endswith(suffix_pattern)]
            elif alt_prefix_pattern is not None and suffix_pattern is None:
                files = [f for f in os.listdir(target_dir) if f.startswith(alt_prefix_pattern)]

        if len(files) > 0 and filter_list is not None:
            files = self._filter(files, filter_list)
        return files
