import datetime
import inspect
import unittest
import utilities
import utilities_dist
import tempfile
import os
import shutil
import sys
import synthpops as sp
from synthpops import cfg

class TestFilePathCreatePop(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.do_plot = False # Whether or not to generate plots
        cls.do_close = False
        cls.dataUSAdir = tempfile.TemporaryDirectory().name
        cls.dataSenegalDir = tempfile.TemporaryDirectory().name
        cls.initial_default_dir = cfg.datadir
        os.makedirs(os.path.join(cls.dataUSAdir, "data"), exist_ok=True)
        os.makedirs(os.path.join(cls.dataSenegalDir, "data"), exist_ok=True)
        cls.subdir_level = "data/demographics/contact_matrices_152_countries"
        cls.sourcedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls.subdir_level)
        patternIgnore1 = ["*contact_networks*", "*contact_networks_facilities*", "*New_York*", "*Oregon*", "*Senegal*"]
        patternIgnore2 = ["*contact_networks*", "*contact_networks_facilities*", "*New_York*", "*Oregon*", "*usa*"]
        utilities.copy_input(cls.sourcedir, cls.dataUSAdir, cls.subdir_level, patterns=patternIgnore1)
        utilities.copy_input(cls.sourcedir, cls.dataSenegalDir, cls.subdir_level, patterns=patternIgnore2)
        cls.n = 10001
        cls.seed = 1
        cls.average_class_size = inspect.signature(sp.make_population).parameters["average_class_size"].default
        cls.average_student_teacher_ratio = inspect.signature(sp.make_population).parameters["average_student_teacher_ratio"].default
        cls.average_student_all_staff_ratio = inspect.signature(sp.make_population).parameters["average_student_all_staff_ratio"].default


    @classmethod
    def copy_output(cls):
        dirname = datetime.datetime.now().strftime("%m%d%Y_%H_%M_%S")
        dirname = os.path.join(os.path.dirname(__file__), dirname)
        os.makedirs(dirname, exist_ok=True)
        for d in [cls.dataUSAdir, cls.dataSenegalDir]:
            for f in os.listdir(d):
                if os.path.isfile(os.path.join(d, f)):
                    shutil.copy(os.path.join(d, f), os.path.join(dirname, f))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.copy_output()
        cfg.set_datadir(cls.initial_default_dir, ["demographics", "contact_matrices_152_countries"])
        cfg.set_location_defaults(country="defaults")
        for d in [cls.dataUSAdir, cls.dataSenegalDir]:
           shutil.rmtree(d, ignore_errors=True)

    def test_usa_location_walk_back(self):
        sp.config.set_datadir(os.path.join(self.dataUSAdir, 'data'), ["demographics", "contact_matrices_152_countries"])
        sp.config.set_alt_location(location="seattle_metro", country_location="usa", state_location="Washington")
        #file_paths = sp.config.FilePaths(location="yakima", country="usa", province="Washington")
        #file_paths.add_alternate_location(location="seattle_metro", country="usa", province="Washington")
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataUSAdir, "data")
        test_prefix = sys._getframe().f_code.co_name
        location = "yakima"
        state_location = "Washington"
        country_location = "usa"
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True,
                                  country_location=country_location, state_location=state_location)
        self.check_result(spop, datadir, self.dataUSAdir, test_prefix, location, state_location, country_location, skip_stat_check=True)

    def test_usa_default_with_param(self):
        sp.config.set_datadir(os.path.join(self.dataUSAdir, 'data'), ["demographics", "contact_matrices_152_countries"])
        sp.config.set_location_defaults(country="usa")
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataUSAdir, "data")
        test_prefix = sys._getframe().f_code.co_name
        location = "seattle_metro"
        state_location = "Washington"
        country_location = "usa"
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True,
                                  country_location=country_location, state_location=state_location)
        self.check_result( spop, datadir, self.dataUSAdir, test_prefix, location, state_location, country_location, skip_stat_check=True)

    def test_usa_default(self):
        sp.config.set_datadir(os.path.join(self.dataUSAdir, 'data'), ["demographics", "contact_matrices_152_countries"])
        sp.config.set_location_defaults(country="usa")
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataUSAdir, "data")
        test_prefix = sys._getframe().f_code.co_name
        location = "seattle_metro"
        state_location = "Washington"
        country_location = "usa"
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True)
        self.check_result( spop, datadir, self.dataUSAdir, test_prefix, location, state_location, country_location, skip_stat_check=True)


    def test_senegal_default(self):
        cfg.set_datadir(os.path.join(self.dataSenegalDir,'data'), ["demographics","contact_matrices_152_countries"])
        cfg.set_location_defaults(country="Senegal")
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataSenegalDir, 'data')
        test_prefix = sys._getframe().f_code.co_name
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True)
        location = cfg.default_location
        state_location = cfg.default_state
        country_location = cfg.default_country
        self.check_result( spop, datadir, self.dataSenegalDir, test_prefix, location, state_location, country_location, skip_stat_check=True)


    def test_senegal_set_param(self):
        cfg.set_datadir(os.path.join(self.dataSenegalDir,'data'), ["demographics","contact_matrices_152_countries"])
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataSenegalDir, 'data')
        test_prefix = sys._getframe().f_code.co_name
        location = "Dakar"
        state_location = "Dakar"
        country_location = "Senegal"
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True,
                                  country_location=country_location, state_location=state_location)
        self.check_result( spop, datadir, self.dataSenegalDir, test_prefix, location, state_location, country_location, skip_stat_check=True)


    def test_senegal_basic(self):
        #everything is default no cfg ops
        rand_seed = self.seed
        n = self.n
        datadir = os.path.join(self.dataSenegalDir, 'data')
        test_prefix = sys._getframe().f_code.co_name
        location = "Dakar"
        state_location = "Dakar"
        country_location = "Senegal"
        spop = sp.make_population(n=n, rand_seed=rand_seed, generate=True,
                                  country_location=country_location, state_location=state_location)
        self.check_result( spop, datadir, self.dataSenegalDir, test_prefix, location, state_location, country_location, skip_stat_check=True)


    def test_senegal_generate_synthetic_population(self):
        # everything is default no cfg ops
        sp.set_seed(self.seed)
        n = self.n
        datadir = os.path.join(self.dataSenegalDir, 'data')
        test_prefix = sys._getframe().f_code.co_name
        location = "Dakar"
        state_location = "Dakar"
        country_location = "Senegal"
        # Note: generate_synthetic_population normally returns None
        #       to get population you need to set return_popdict to True
        spop = sp.generate_synthetic_population(n=n, datadir=datadir, location=location, country_location=country_location, state_location=state_location, return_popdict=True)

        self.check_result( spop, datadir, self.dataSenegalDir, test_prefix, location, state_location, country_location, skip_stat_check=True)

    def check_result(self, pop, datadir, figdir, test_prefix, location, state_location, country_location, skip_stat_check=False):
        if self.do_plot:
            utilities.check_class_size(pop, self.average_class_size, self.average_student_teacher_ratio,self.average_student_all_staff_ratio, 1)
            utilities.check_teacher_staff_ratio(pop, datadir, f"{test_prefix}", self.average_student_teacher_ratio,
                                                self.average_student_all_staff_ratio, err_margin=2)
            utilities_dist.check_age_distribution(pop, self.n, datadir, figdir, location, state_location, country_location,
                                             test_prefix=test_prefix, do_close=self.do_close)
            utilities_dist.check_enrollment_distribution(pop, self.n, datadir, figdir, location, state_location, country_location,
                                                    test_prefix=f"{test_prefix}", skip_stat_check=skip_stat_check, do_close=self.do_close)


if __name__ == "__main__":
    unittest.main()