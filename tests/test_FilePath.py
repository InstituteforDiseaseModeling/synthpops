import datetime
import inspect
import unittest
import utilities
import tempfile
import os
import shutil
import sys
import synthpops as sp
from synthpops import cfg
#import sciris as sc
import pytest


class TestFilePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rootDir = tempfile.TemporaryDirectory().name
        cls.initia_default_dir = cfg.datadir
        os.makedirs(os.path.join(cls.rootDir, "data"), exist_ok=True)
        cls.dataDir = os.path.join(cls.rootDir, "data")

    @classmethod
    def tearDownClass(cls) -> None:
        #cls.copy_output()
        cfg.set_datadir(cls.initia_default_dir, ["demographics","contact_matrices_152_countries"])
        cfg.set_location_defaults(country="default")
        #for d in [cls.rootDir]:
        #    shutil.rmtree(d, ignore_errors=True)

    def test_FilePath_init_None(cls):
        with pytest.raises(NotImplementedError):
            file_paths = sp.config.FilePaths()

    def test_check_defaults(cls):
        # defaults ae set when config.py is loaded. check for the defaults
        # should be the same as usa
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'

    def test_FilePath_no_country(cls):
        with pytest.raises(NotImplementedError):
            file_paths = sp.config.FilePaths('seattle_metro', 'Washington')

    def test_set_location_defaults_none(cls):
        sp.config.set_location_defaults()
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'

    def test_set_location_defaults_Senegal(cls):
        sp.config.set_location_defaults('Senegal')
        assert sp.config.default_country == 'Senegal'
        assert sp.config.default_state == 'Dakar'
        assert sp.config.default_location == 'Dakar'
        assert sp.config.default_sheet_name is None

    def test_set_location_defaults_usacls(cls):
        sp.config.set_location_defaults('usa')
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'


    def test_FilePath_country_only_usa(cls):
        file_paths = sp.config.FilePaths(country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is None
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    def test_FilePath_country_only_Senegal(cls):
        file_paths = sp.config.FilePaths(country='Senegal')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'Senegal'
        assert file_paths.province is None
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    def test_FilePath_invalid_country(cls):
        with pytest.raises(FileNotFoundError):
            file_paths = sp.config.FilePaths(country='Germany')

    # Test valid country state combinations
    def test_FilePath_usa_Washington(cls):
        file_paths = sp.config.FilePaths(province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 2

    def test_FilePath_Senegal_Dakar(cls):
        file_paths = sp.config.FilePaths(province='Dakar', country='Senegal')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'Senegal'
        assert file_paths.province is not None
        assert file_paths.province == 'Dakar'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 2

    # invalid state
    def test_FilePath_usa_Florida(cls):
        file_paths = sp.config.FilePaths(province='Florida', country='usa')
        # note because usa is valid and Florida is no, searches will be at USA level
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Florida'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    # test valid locations
    def test_FilePath_usa_Washington_seattle_metro(cls):
        file_paths = sp.config.FilePaths(location='seattle_metro', province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is not None
        assert file_paths.location == 'seattle_metro'
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 3

    def test_set_alt_location(cls):
        current_alt_location = sp.config.alt_location
        sp.config.set_alt_location('Yakima', 'Washington', 'usa')
        assert sp.config.alt_location.country_location == 'usa'
        assert sp.config.alt_location.state_location == 'Washington'
        assert sp.config.alt_location.location == 'Yakima'
        sp.config.alt_location = current_alt_location

    def test_FilePaths_usa_Washington_yakima_with_alt_location(cls):
        current_alt_location = sp.config.alt_location
        sp.config.set_alt_location(location='seattle_metro', state_location='Washington', country_location='usa')
        file_paths = sp.config.FilePaths(location='yakima',province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is not None
        assert file_paths.location == 'yakima'
        assert file_paths.alt_country is not None
        assert file_paths.alt_country == 'usa'
        assert file_paths.alt_province is not None
        assert file_paths.alt_province == 'Washington'
        assert file_paths.alt_location is not None
        assert file_paths.alt_location == 'seattle_metro'
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 5
        sp.config.alt_location = current_alt_location

    # file walk back test
    # We assume:
    # 1) that there is a directory structure .../country[/province[/location]]/data_type
    # 2) that files are geo referenced  (e.g. mycity_myfile...) or fixed (e.g. myfile)
    # 3) The initial search can return multiple files
    # 4) The first file that meets the qualification is returned
    # Note at the lower functions we do not check data type

    def create_test_dirs(cls, dir, relpath, country, state, location, test_id, geo_locate=True, walkback=False, walkback_level=0, altfile=False, alt_level=0):
        # used to create the test directories
        #   dir is the same as datadir,
        #   relpath is the same as the relpat used to set up datadir
        #   country - the country
        #   state - state
        #   location - the location
        #   test_id use in the file name and what is written in the file
        #   geo_locate - append the name of the level to the file
        #   walkback - if true setup a walkback test
        #   walkback_level -  what level to write the file (0 = country, 1=state, 2 = location)
        #   altfile - setup altfile test
        #   alt_level - what level to write the file (0 = country, 1=state, 2 = location)
        path = dir
        path = dir
        suffix = '.dat'
        if relpath != None and len(relpath) > 0:
            # create relative path
            path = os.path.join(dir, *relpath)
            os.makedirs(path, exist_ok=True)
            #make the needed sub directories and files
        place_dir = path
        place_tree = [country, state, location]
        level = 0

        for place in place_tree:
            if place is not None:
                place_dir = os.path.join(place_dir, place)
                os.makedirs(os.path.join(place_dir, test_id), exist_ok=True)
                places = []
                if geo_locate == True:
                    places.append(place)
                if walkback and walkback_level == level:
                    places.append('walkback')
                places.append(test_id)
                place_file = '_'.join(places)
                place_file = place_file + suffix

                place_path = os.path.join(place_dir, test_id, place_file)
                if altfile:
                    if alt_level == level:
                        with(open( place_path, 'w')) as f:
                            print(f"test:{test_id} file for location {place}, {place_file}", file=f)
                else:
                    with(open( place_path, 'w')) as f:
                        print(f"test:{test_id} file for location {place}, {place_file}", file=f)
            level +=1


    # test getting a valid file at level
    def test_get_dat_file_at_location(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'test1'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = location + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {location}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(location, state, country)
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]
        assert results == target_line

    def test_get_dat_file_at_state(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'test2'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = state + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {state}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(None, state, country)
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line


    def test_get_dat_file_at_country(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'test3'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = country+ '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {country}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(None, None, country)
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line


    # test getting a valid file at level
    def test_get_dat_file_walkback_at_state(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'test4'
        prefix = '{location}_walkback_' + test_id
        suffix = '.dat'
        target_file = state + '_walkback_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {state}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, walkback=True,walkback_level=1)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(location, state, country)
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]
        assert results == target_line

    def test_get_demographic_file_at_location(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'age_distributions'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = location + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {location}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(location, state, country)
        file = file_paths.get_demographic_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]
        assert results == target_line

    def test_get_demographic_file_at_state(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'age_distributions'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = state + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {state}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(None, state, country)
        file = file_paths.get_demographic_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path

        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line

    def test_get_demographic_file_walkback_at_state(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        test_id = 'age_distributions'
        prefix = '{location}_walkback_' + test_id
        suffix = '.dat'
        target_file = state + '_walkback_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {state}, {target_file}\n"

        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, walkback=True,walkback_level=1)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(location, state, country)
        file = file_paths.get_demographic_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        assert file is not None
        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]
        assert results == target_line

    # test for alternate locations

    def create_alt_test_dirs(cls, dir, relpath, country, state, location, test_id, prefix,   file_level=0):
        # create alternate structure for test.
        #   dir is the same as datadir,
        #   relpath is the same as the relpat used to set up datadir
        #   country - the country
        #   state - state
        #   location - the location
        #   test_id use in the file name and what is written in the file
        #   prefix - the prefix to attach to the file
        #   file_level - what level to write the file (0 = country, 1=state, 2 = location)
        path = dir
        suffix = '.dat'
        if relpath != None and len(relpath) > 0:
            # create relative path
            path = os.path.join(dir, *relpath)
            os.makedirs(path, exist_ok=True)
            #make the needed sub directories and files
        place_dir = path
        place_tree = [country, state, location]
        level = 0
        file_name = prefix + '_' + test_id + suffix
        for place in place_tree:
            if place is not None:
                place_dir = os.path.join(place_dir, place)
                os.makedirs(os.path.join(place_dir, test_id), exist_ok=True)
                file_path = os.path.join(place_dir, test_id, file_name)
                if  file_level == level:
                    with(open( file_path, 'w')) as f:
                            print(f"test:{test_id} file for location {place}, {file_name}", file=f)
            level +=1

    def test_get_dat_file_alt_location(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        alt_location = 'fum'
        test_id = 'test8'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = alt_location + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_location}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id,geo_locate=False)
        cls.create_test_dirs(cls.dataDir, rel_path, country, state, alt_location, test_id, altfile=True, alt_level=2)
        cfg.set_datadir(cls.dataDir, [])

        file_paths = cfg.FilePaths(location, state, country)
        file_paths.add_alternate_location(location=alt_location, province = state, country=country)
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line

    def test_get_dat_file_alt_location_using_set_altlocation(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        current_alt_location = cfg.alt_location

        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        alt_location = 'fum'
        test_id = 'test9'
        prefix = '{location}_' + test_id
        suffix = '.dat'
        target_file = alt_location + '_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_location}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        cls.create_test_dirs(cls.dataDir, rel_path, country, state, location, test_id,geo_locate=False)
        cls.create_test_dirs(cls.dataDir, rel_path, country, state, alt_location, test_id, altfile=True, alt_level=2)
        cfg.set_datadir(cls.dataDir, [])
        cfg.set_alt_location(location=alt_location, state_location=state, country_location=country)

        file_paths = cfg.FilePaths(location, state, country)

        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        cfg.alt_location = current_alt_location

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line

    def test_get_dat_file_alt_location_with_alt_prefix(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        current_alt_location = cfg.alt_location

        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'foo'
        alt_location = 'fum'
        test_id = 'test10'
        prefix = 'Yakima'
        alt_prefix = 'seattle_metro'
        suffix = '.dat'
        target_file = 'seattle_metro_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_location}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, prefix=prefix,file_level=3 )
        cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, alt_location, test_id, prefix=alt_prefix,file_level=2)
        cfg.set_datadir(cls.dataDir, [])
        cfg.set_alt_location(location=alt_location, state_location=state, country_location=country)

        file_paths = cfg.FilePaths(location, state, country)
        file_prefix = prefix + '_' + test_id
        alt_firl_prefix = alt_prefix + '_' + test_id
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix, alt_prefix = alt_prefix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        cfg.alt_location = current_alt_location

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line


    def test_get_dat_file_alt_location_with_alt_prefix_no_dirs(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        current_alt_location = cfg.alt_location
        # similar test to the previous test, only this time we are not
        # going to create the primary directory for 'bar', just the dir fum

        rel_path = None

        country = 'fee'
        state   = 'fi'
        location= 'bar'
        alt_location = 'fum'
        test_id = 'test11'
        prefix = 'Yakima'
        alt_prefix = 'seattle_metro'
        suffix = '.dat'
        target_file = 'seattle_metro_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_location}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        #cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, prefix=prefix,file_level=3 )
        cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, alt_location, test_id, prefix=alt_prefix,file_level=2)
        cfg.set_datadir(cls.dataDir, [])
        cfg.set_alt_location(location=alt_location, state_location=state, country_location=country)

        file_paths = cfg.FilePaths(location, state, country)
        file_prefix = prefix + '_' + test_id
        alt_firl_prefix = alt_prefix + '_' + test_id
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix, alt_prefix = alt_prefix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        cfg.alt_location = current_alt_location

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line


    def test_get_dat_file_alt_location_with_alt_prefix_no_Or(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        current_alt_location = cfg.alt_location
        # similar test 11 , only this time we are not
        # going to create the primary directory for 'for Origon', just the dir fum

        rel_path = None

        country = 'usa'
        state   = 'Origon'
        location= 'Portland'
        alt_state = 'Washington'
        alt_location = 'fum'
        test_id = 'test11'
        prefix = 'Portland'
        alt_prefix = 'seattle_metro'
        suffix = '.dat'
        target_file = 'seattle_metro_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_state}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        #cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, prefix=prefix,file_level=3 )
        cls.create_alt_test_dirs(cls.dataDir, rel_path, country, alt_state, alt_location, test_id, prefix=alt_prefix,file_level=1)
        cfg.set_datadir(cls.dataDir, [])
        cfg.set_alt_location(location=alt_location, state_location=alt_state, country_location=country)

        file_paths = cfg.FilePaths(location, state, country)
        file_prefix = prefix + '_' + test_id
        alt_firl_prefix = alt_prefix + '_' + test_id
        file = file_paths.get_data_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix, alt_prefix = alt_prefix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        cfg.alt_location = current_alt_location

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]
        assert results == target_line

    def test_get_demographic_file_alt_location_with_alt_prefix_no_Or(cls):
        #get current dataDir and rel_path
        current_dataDir = cfg.datadir
        current_rel_path = cfg.rel_path
        current_alt_location = cfg.alt_location
        # similar test 11 , only this time we are not
        # going to create the primary directory for 'for Origon', just the dir fum

        rel_path = None

        country = 'usa'
        state   = 'Origon'
        location= 'Portland'
        alt_state = 'Washington'
        alt_location = 'fum'
        test_id = 'assisted_living'
        prefix = 'Portland'
        alt_prefix = 'seattle_metro'
        suffix = '.dat'
        target_file = 'seattle_metro_' + test_id + '.dat'
        target_line =f"test:{test_id} file for location {alt_state}, {target_file}\n"

        # we are testing to see if we cans skip file in the primary and find it in the secondary
        # to do this we will create files in all directories, for the primary we will not
        # geo-locate, and for the alt_we will. then search for a Geo-located file
        #cls.create_alt_test_dirs(cls.dataDir, rel_path, country, state, location, test_id, prefix=prefix,file_level=3 )
        cls.create_alt_test_dirs(cls.dataDir, rel_path, country, alt_state, alt_location, test_id, prefix=alt_prefix,file_level=1)
        cfg.set_datadir(cls.dataDir, [])
        cfg.set_alt_location(location=alt_location, state_location=alt_state, country_location=country)

        file_paths = cfg.FilePaths(location, state, country)
        file_prefix = prefix + '_' + test_id
        alt_firl_prefix = alt_prefix + '_' + test_id
        file = file_paths.get_demographic_file(location=location, filedata_type=test_id, prefix=prefix, suffix=suffix, alt_prefix = alt_prefix)
        cfg.data_dir = current_dataDir
        cfg.rel_path = current_rel_path
        cfg.alt_location = current_alt_location

        assert file is not None

        with (open(file,'r')) as file:
            test_line = file.readlines()
            results = test_line[0]

        assert results == target_line



