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
import datetime
import sciris as sc
import re
import yaml

__all__ = ['logger', 'checkmem', 'datadir', 'localdatadir', 'rel_path', 'alt_rel_path', 'set_datadir',  'set_nbrackets', 'validate', 'set_altdatadir',
           'set_location_defaults', 'default_country', 'default_state', 'default_location', 'default_sheet_name',  'alt_location', 'default_household_size_1_included']


class LocationClass:
    def __init__(self, location=None, state_location=None, country_location=None):
        self.country_location = country_location
        self.state_location = state_location
        self.location = location

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

#%% Logger -- adapted from Atomica

# Set the default logging level
default_log_level = ['DEBUG', 'INFO', 'WARNING', 'CRITICAL'][1]

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


#%% Functions
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
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        if 'valid_nbrackets' in data.keys():
            valid_nbracket_ranges = data['valid_nbrackets']

        if country_location in data.keys():
            loc = data[country_location]
            default_location = loc['location']
            default_state = loc['province']
            default_country = loc['country']
            default_sheet_name = loc['sheet_name']
            nbrackets = 20 if loc['nbrackets'] is None else loc['nbrackets']
            default_household_size_1_included = False if 'household_size_1' not in loc.keys() else loc['household_size_1']

        else:
            logger.warning(f"warning: country not in config file, using defaults")
            loc = data['defaults']
            default_location = loc['location']
            default_state = loc['province']
            default_country = loc['country']
            default_sheet_name = loc['sheet_name']
            nbrackets = 20 if 'nbrackets' not in loc.keys() else loc['nbrackets']
            default_household_size_1_included = False if 'household_size_1' not in loc.keys() else loc['household_size_1']


def set_alt_location(location=None, state_location=None, country_location=None):
    global alt_location
    levels = [location,state_location, country_location]

    if all(level is None for level in levels) :
        alt_location = None
        logger.warning(f"Warning: No alternate location specified")
    elif country_location is None:
        alt_location = None
        logger.warning(f"Warning: No alternate country specified, alternate country is required")
    elif country_location is not None and state_location is None:
        # ifstate_location is none make sure alt_location only has country
        alt_location = LocationClass(country_location=country_location)
    else:
        alt_location = LocationClass(location=location, state_location=state_location, country_location=country_location)


set_location_defaults()

logger.info(f'Loading SynthPops: {thisdir}')
logger.debug(f'Data folder: {datadir}')


def set_datadir(root_dir, relative_path=None):
    '''Set the data folder and relative path to the user-specified
        location.
        On startup, the datadir and rel_path are set to the conventions
        used to store data. datadir is the root directory to the data, and
        rel_path is a list of sub directories to the data -->
        rel_path = ['demographics', 'contact_matrices_152_countries']
        to change the location of the data the user is able to supply a new root_dir and new relative path. If the user uses a similar directory path model that we use
        e.g. root_dir/demographic/contact... the user can change datadir without changing relitive path, by passing in relative_path = None (default)
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


def set_altdatadir(root_dir, relative_path=None):
    '''Set the data folder to the user-specified location -- note, mostly deprecated.'''
    global alt_datadir
    global alt_rel_path
    alt_datadir = root_dir
    if relative_path is not None:
        alt_rel_path = relative_path
    logger.info(f'Done: alt data directory set to {root_dir}.')
    logger.info(f'alt relative Path set to  {rel_path}.')
    return alt_datadir


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

"""
    It is assumed that the data files used for synthpops are organized by location path
    and data type. A location path consisting of multiple administrative levels commonly
    referred to as country, state/province and and location.Location can be a district
    (w.g. Kin County), city (Seattle) or a metropolitan area (e.g. Seattle metropolitan area).
    Each of these levels is represents by a directory in the data tree and contains the data
    for administrative boundaries

    When storing the data files, each administrative area is represented by a directory.
    In addition to each administrative directory there is a list of optional parameter directories,
    and each parameter directory contains and actual data file. An example of a data storage
    structure is described below.
        /data_dir/ --+
                     |
                     + country 1 +
                     |           + province_a +
                     |           |            + location_1 +
                     |           |            |            + age_distribution
                     |           |            |            + contact_networks
                     |           |            |            + employment
                     |           |            |            + enrollment
                     |           |            |            + household_living_arrangements
                     |           |            |            + household size distribution
                     |           |            |            + schools
                     |           |            |            + workplaces
                     |           |            + location_2
                     |           |            |            + age_distribution
                     |           |            |            + contact_networks
                     |           |            |            + employment
                     |           |            |            + enrollment
                     |           |            |            + household_living_arrangements
                     |           |            |            + household size distribution
                     |           |            |            + schools
                     |           |            |            + workplaces
                     |           |            + age_distribution
                     |           |            + contact_networks
                     |           |            + employment
                     |           |            + enrollment
                     |           |            + household_living_arrangements
                     |           |            + household_size_distribution
                     |           |            + schools
                     |           |            + workplaces
                     |           + age_distribution
                     |           + contact_networks
                     |           + employment
                     |           + enrollment
                     |           + household_living _arrangements
                     |           + household_size_distribution
                     |           + schools
                     |           + workplaces
                     + defaults +
                     |          + age_distribution
                     |          + contact_networks
                     |          + employment
                     |          + enrollment
                     |          + household_living_arrangements
                     |          + household_size_distribution
                     |          + schools
                     |          + workplaces
    Data data is not always availability at the highest resolution (e.g. location level).
    And when that occurs, we want to fall back to the next highest level. For example,
    if location_1 has not age_distribution data (represented by an empty directory or
    no directory) under location_1, then we want to see if there is and age_distributon
    data file in province+_a. If there is no age_distribution in province_a, then we want to
    use the age_distribution in country_1.

    Alternate location:
    It is not always possible to get all the for all country provinces or location. Because
    of this we allow for the use of 'similar data'. An example of this would be location like
    Portland and Seattle. Population size, age distribution and industrial mix are very similar.
    If we were missing household or school information for Portland, it reasonable to use
    Seattle data for these items. This class allows the user to define an alternate location
    to search for missing data.

    Default Location:
    Finally if the 'use_default' flag is set True, and data is not found the default data will be
    used.

    Assumption:
    1) We always want the data for the  highest resolution on the primary location path
    2) If the data is not found in the primary location path, the alternate path will be
    searched
    3) By default the default path is not searched
    4) if the data  is not found on the primary or alternate path an error is thrown.
"""


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

    def add_alternate_location(self, location=None, province=None, country=None):
        levels = [location,province, country]

        altdirs = None
        if all(level is None for level in levels) :
            self.alt_country = None
            self.alt_province = None
            self.alt_location = None
            logger.warning(f"Warning: No alternate location specified")
        elif country is None:
            self.alt_country = None
            self.alt_province = None
            self.alt_location = None
            logger.warning(f"Warning: No alternate country specified, alternate country is required")
        elif country is not None and province is None:
            # if province is none make sure location is none
            self.alt_country = country
            self.alt_province = None
            self.alt_location = None
            altdirs = self._add_dirs(self.alt_root_dir, None, None,country)
        else:
            self.alt_country = country
            self.alt_province = province
            self.alt_location = location
            # build alternate dirs
            altdirs = self._add_dirs(self.alt_root_dir, location, province,country)

        if len(altdirs) > 0:
            altdirs.reverse()
            self.basedirs.extend(altdirs)
        self.validate_dirs()

    def _add_dirs(self, root,location, province, country):
        levels = [location,province, country]
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



    @staticmethod
    def norm_join(dir, path):
        return os.path.normpath(os.path.join(dir, path))

    def get_dir_list(self):
        return self.basedirs

    def get_location(self):
        location_info = LocationClass(location=location, state_locaitno=province, country_location=country)
        return location_info

    def get_alt_location(self):
        location_info = LocationClass(location=atl_location, state_location=alt_province, country_location=alt_country)
        return location_info

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

    def _filter(self, file_list, filter_list):
        # the filter is a list of numbers added to the file name representing the number
        # of brackets in the file (e.g. age brackets). We want the highest number possible
        # get only the files in the list
        # assume the filter is a list of numbers, and file name ins in .dat
        def extract_number(f):
            s = re.findall(r"\d+", f)
            # return (int(s[0]) if s else -1,f)
            return (int(s[0]) if s else -1)

        #files = []
        #if len(str_list ) > 0:
        #    files = [max(str_list, key=extract_number)]
        #files = [max(file_list, key=extract_number)]
        files = [f for f in file_list if extract_number(f) == filter_list]
        return files
