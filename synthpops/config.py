'''
This module sets the location of the data folder.
'''

#%% Housekeeping

import os
import datetime
import sciris as sc
import re
import yaml

__all__ = ['datadir', 'localdatadir', 'rel_path', 'set_datadir', 'set_nbrackets', 'validate', 'set_altdatadir',
           'set_location_defaults', 'default_country', 'default_state', 'default_location', 'default_sheet_name']

# Declaring this here makes it globally available as synthpops.datadir
datadir = None
alt_datadir = None
localdatadir = None
rel_path = ['demographics', 'contact_matrices_152_countries']
full_data_available = False # this is likely not necesary anymore

# Set the local data folder
#thisdir = sc.thisdir(__file__)
thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
config_file = os.path.join(thisdir, 'config_info.yaml')
#localdatadir = os.path.abspath(os.path.join(thisdir, os.pardir, 'data'))
#recomended change
localdatadir = os.path.abspath(os.path.join(thisdir, os.pardir, 'data'))


# Replace with local data dir if Dropbox folder is not found
if datadir is None:
    full_data_available = True
    datadir = localdatadir


# Number of census age brackets to use
# added 18 to support Senegal
nbrackets = [16, 18, 20][1] # Choose how many age bins to use -- 20 is only partially supported
matrix_size = 16 # The dimensions of the mixing matrices -- currently only 16 is available
default_country = None
default_state = None
default_location = None
default_sheet_name = "United States of America"

#%% Functions
def set_location_defaults(country=None):
    global config_file
    global default_country
    global default_state
    global default_location
    global default_sheet_name

    # read the yaml file
    country_location = country if country is not None else 'defaults'
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        if country_location in data.keys():
            loc = data[country_location]
            default_location = loc[0]
            default_state = loc[1]
            default_country =loc[2]
        else:
            print(f"warning: country not in config file, using defaults")
            loc = data['defaults']
            default_location = loc[0]
            default_state = loc[1]
            default_country =loc[2]

set_location_defaults()

def set_datadir(folder):
    '''Set the data folder to the user-specified location -- note, mostly deprecated.'''
    ''' user specifies complete path'''
    global datadir
    global rel_path
    datadir = folder
    rel_path = []
    print(f'Done: data directory set to {folder}.')
    return datadir

def set_altdatadir(folder):
    '''Set the data folder to the user-specified location -- note, mostly deprecated.'''
    global alt_datadir
    alt_datadir = folder
    print(f'Done: data directory set to {folder}.')
    return alt_datadir

def set_nbrackets(n):
    '''Set the number of census brackets -- usually 16 or 20.'''
    global nbrackets
    print(f"set_nbrackets n = {n}")
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
                     |           |            |            + household living arrangements
                     |           |            |            + household size distribution
                     |           |            |            + schools
                     |           |            |            + workplaces
                     |           |            + location_2
                     |           |            |            + age_distribution
                     |           |            |            + contact_networks
                     |           |            |            + employment
                     |           |            |            + enrollment
                     |           |            |            + household living arrangements
                     |           |            |            + household size distribution
                     |           |            |            + schools
                     |           |            |            + workplaces
                     |           |            + age_distribution
                     |           |            + contact_networks
                     |           |            + employment
                     |           |            + enrollment
                     |           |            + household living arrangements
                     |           |            + household size distribution
                     |           |            + schools
                     |           |            + workplaces
                     |           + age_distribution
                     |           + contact_networks
                     |           + employment
                     |           + enrollment
                     |           + household living arrangements
                     |           + household size distribution
                     |           + schools
                     |           + workplaces
                     + defaults +
                     |          + age_distribution
                     |          + contact_networks
                     |          + employment
                     |          + enrollment
                     |          + household living arrangements
                     |          + household size distribution
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
class FilePaths():
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

    def __init__(self,  location=None, province=None, country=None,  alt_location= None, alt_province=None, alt_country=None, root_dir=None, alt_rootdir=None,  use_defaults=False):
        global datadir, alt_datadir, rel_path
        base_path = datadir
        if len(rel_path) > 0:
            base_dir= os.path.join(datadir, *rel_path)
        self.root_dir = base_dir if root_dir is None else root_dir

        self.alt_root_dir = alt_datadir if alt_rootdir is None else alt_rootdir
        self.country = None
        self.province = None
        self.location = None
        self.alt_country = None
        self.alt_province = None
        self.alt_location = None

        if self.alt_root_dir is None:
            self.alt_root_dir = self.root_dir
        self.basedirs = []

        self.add_base_location(location, province, country)

        self.add_alternate_location(alt_location, alt_province, alt_country)

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

        if all(level is None for level in levels) :
            self.alt_country = None
            self.alt_province = None
            self.alt_location = None
            #print(f"Warning: No alternate location specified")
        elif country is None:
            self.alt_country = None
            self.alt_province = None
            self.alt_location = None
            #print(f"Warning: No alternate country specified, alternate country is required")
        else:
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
        for i,e in reversed(list(enumerate(self.basedirs))):
            if not os.path.isdir(e[1]):
                print(f"Warning: Directory {e[1]} missing, removing.")
                self.basedirs.pop(i)
        # make sure we have at least one directory, or through an error
        if len(self.basedirs) < 1:
             raise FileNotFoundError(f'The location data folders do not exist, as far as I can tell.')
             return False
        return True



    @staticmethod
    def norm_join(dir, path):
        return os.path.normpath(os.path.join(dir, path))

    def get_dir_list(self):
        return self.basedirs

    def get_location(self):
        location_info = []
        location_info.append(self.location)
        location_info.append(self.province)
        location_info.append(self.country)
        return location_info

    def get_alt_location(self):
        location_info = []
        location_info.append(self.alt_location)
        location_info.append(self.alt_province)
        location_info.append(self.alt_country)
        return location_info

    def get_demographic_file(self, location=None, filedata_type=None, prefix=None, suffix=None, filter_list=None):
        """
        Search the base directories and return the first file found that matches the criteria
        """
        filedata_types = ['age_distributions', 'assisted_living', 'contact_networks', 'employment', 'enrollment', 'household_living_arrangements', 'household_size_distributions', 'schools', 'workplaces']
        if filedata_type is None:
            raise NotImplementedError(f"Missing filedata_type string.")
            return None

        if filedata_type not in filedata_types:
            raise NotImplementedError(f"Invalid filedata_type string {filedata_type}. filedata_type must be one of the following {filedata_types}")
            return None

        file = self._search_dirs(location, filedata_type, prefix, suffix, filter_list)
        return file

    def get_data_file(self, location=None, prefix=None, suffix=None, filter_list=None):
        """
        Search the base directories and return the first file found that matches the criteria
        """
        filedata_type = None
        file = self._search_dirs(location, filedata_type, prefix, suffix, filter_list)
        return file

    def _search_dirs(self, location, filedata_type, prefix, suffix, filter_list):
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
                    files = self._list_files(target_location, filedata_dir, prefix, suffix, filter_list)

                    if len(files) > 0:
                        results = os.path.join(filedata_dir, files[0])
                        break
                else:
                    print(f'no data in directory {filedata_dir}, skipping')
        return results

    def _list_files(self,level, target_dir,  prefix, suffix, filter_list):
        global nbrackets
        prefix_pattern = prefix
        suffix_pattern = suffix
        if level is not None and prefix_pattern is not None:
            prefix_pattern = prefix_pattern.format(location=level)

        #if level is not None and suffix is not None:
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

        if len(files) > 0 and filter_list is not None:
            files = self._filter(files, filter_list)
        return files

    def _filter(self,file_list,filter_list):
        # the filter is a list of numbers added to the file name representing the number
        # of brackets in the file (e.g. age brackets). We want the highest number possible
        # get only the files in the list
        # assume the filter is a list of numbers, and file name ins in .dat
        def extract_number(f):
            s = re.findall("\d+",f)
            #return (int(s[0]) if s else -1,f)
            return (int(s[0]) if s else -1)

        #files = []
        #if len(str_list ) > 0:
        #    files = [max(str_list, key=extract_number)]
        #files = [max(file_list, key=extract_number)]
        files = [f for f in file_list if extract_number(f) == filter_list]
        return files


