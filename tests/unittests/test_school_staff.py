"""
test school staff features
"""
import unittest
import tempfile
import os
import sys
import shutil
import datetime
import synthpops as sp
import sciris as sc
import utilities
import utilities_dist
from synthpops import config as cfg


class TestSchoolStaff(unittest.TestCase):

    do_close = True # Whether or not to close plots after saving them to disk
    n = 5001 # Number of people to run the tests for

    @classmethod
    def setUpClass(cls) -> None:
        cls.do_plot = False
        cls.resultdir = tempfile.TemporaryDirectory().name
        cls.dataDir = os.path.join(cls.resultdir, "data")
        cls.subdir_level = os.path.join("data")
        cls.sourcedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir, cls.subdir_level)
        utilities.copy_input(cls.sourcedir, cls.resultdir, cls.subdir_level)
        cfg.set_nbrackets(20)
        cfg.set_location_defaults('defaults')

    @classmethod
    def tearDownClass(cls) -> None:
        # cls.copy_output()
        shutil.rmtree(cls.resultdir, ignore_errors=True)

    @classmethod
    def copy_output(cls):
        dirname = datetime.datetime.now().strftime("%m%d%Y_%H_%M")
        dirname = os.path.join(os.path.dirname(__file__), dirname)
        os.makedirs(dirname, exist_ok=True)
        for f in os.listdir(cls.resultdir):
            if os.path.isfile(os.path.join(cls.resultdir, f)):
                shutil.copy(os.path.join(cls.resultdir, f), os.path.join(dirname, f))

    @unittest.skip("this long running scenario is excluded from BVT (basic verification tests)")
    def test_scale(self):
        seed = 1
        # set param
        average_student_teacher_ratio = 22
        average_student_all_staff_ratio = 15
        datadir = self.dataDir
        location = 'seattle_metro'
        state_location = 'Washington'
        country_location = 'usa'
        i = 0
        for n in [2001, 10001]:
            try:
                pop = {}
                sp.set_seed(seed)
                print(seed)
                pop = sp.generate_synthetic_population(n, datadir,average_student_teacher_ratio=average_student_teacher_ratio,
                                                       average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                       return_popdict=True)
                sc.savejson(os.path.join(self.resultdir, f"calltwice_{n}_{i}.json"), pop, indent=2)
                result = utilities.check_teacher_staff_ratio(pop, self.dataDir, f"calltwice_{n}_{i}", average_student_teacher_ratio,
                                                             average_student_all_staff_ratio=average_student_all_staff_ratio, err_margin=2)
                utilities_dist.check_enrollment_distribution(pop, n, datadir, location, state_location, country_location,
                                                        test_prefix=f"calltwice{n}_{i}", skip_stat_check=True, do_close=self.do_close)
                utilities_dist.check_age_distribution(pop, n, datadir, self.resultdir, location, state_location, country_location,
                                                 test_prefix=f"calltwice{n}_{i}", do_close=self.do_close)
                i += 1
            except:
                print("check failed, continue...")
        return result


    def test_staff_generate(self):

        """
        generate a population and check if teacher/staff ratio match
        """
        rand_seed = 1
        sp.set_seed(rand_seed)
        #set param
        n = self.n
        datadir = self.dataDir
        location = 'seattle_metro'
        state_location = 'Washington'
        country_location = 'usa'
        sheet_name = 'United States of America'
        with_school_types = False
        school_mixing_type = 'random'
        average_class_size = 20
        inter_grade_mixing = 0.1
        average_student_teacher_ratio = 20
        average_teacher_teacher_degree = 3
        teacher_age_min = 25
        teacher_age_max = 75
        average_student_all_staff_ratio = 12
        average_additional_staff_degree = 18
        staff_age_min = 20
        staff_age_max = 75
        return_popdict = True
        test_prefix = sys._getframe().f_code.co_name
        vals = locals()
        pop = utilities.runpop(resultdir=self.resultdir, testprefix=f"{test_prefix}", actual_vals=vals, method=sp.generate_synthetic_population)
        if self.do_plot:
            utilities.check_class_size(pop, average_class_size, average_student_teacher_ratio,
                                           average_student_all_staff_ratio, 1)
            result = utilities.check_teacher_staff_ratio(pop, self.dataDir, f"{test_prefix}", average_student_teacher_ratio, average_student_all_staff_ratio, err_margin=2)
            utilities_dist.check_age_distribution(pop, self.n, datadir, self.resultdir, location, state_location, country_location, test_prefix=test_prefix, do_close=self.do_close)
            utilities_dist.check_enrollment_distribution(pop, self.n, datadir, self.resultdir, location, state_location, country_location, test_prefix=f"{test_prefix}", do_close=self.do_close)


    def test_with_ltcf(self):
        """
        test with long term care facilities options
        """
        rand_seed = 1
        sp.set_seed(rand_seed)
        # set param
        n = self.n
        datadir = self.dataDir
        location = 'seattle_metro'
        state_location = 'Washington'
        country_location = 'usa'
        sheet_name = 'United States of America'
        use_two_group_reduction = True
        average_LTCF_degree = 20
        ltcf_staff_age_min = 20
        ltcf_staff_age_max = 65
        with_school_types = True
        average_class_size = 20
        inter_grade_mixing = 0.1
        average_student_teacher_ratio = 20.0
        average_teacher_teacher_degree = 3
        teacher_age_min = 25
        teacher_age_max = 70

        with_non_teaching_staff = True
        average_student_all_staff_ratio = 11
        average_additional_staff_degree = 20
        staff_age_min = 20
        staff_age_max = 75
        school_mixing_type = {'pk': 'age_and_class_clustered', 'es': 'random', 'ms': 'age_clustered', 'hs': 'random', 'uv': 'random'}
        return_popdict = True
        vals = locals()
        test_prefix = sys._getframe().f_code.co_name
        pop = utilities.runpop(resultdir=self.resultdir, testprefix=test_prefix, actual_vals=vals,
                               method=None)
        if self.do_plot:
            utilities.check_class_size(pop, average_class_size, average_student_teacher_ratio,
                                       average_student_all_staff_ratio, 1)
            result = utilities.check_teacher_staff_ratio(pop, datadir, test_prefix, average_student_teacher_ratio,
                                                         average_student_all_staff_ratio, err_margin=2)
            utilities_dist.check_age_distribution(pop, self.n, datadir, self.resultdir, location, state_location, country_location, test_prefix=test_prefix, do_close=self.do_close)
            utilities_dist.check_enrollment_distribution(pop, self.n, datadir, self.resultdir, location, state_location, country_location, test_prefix=test_prefix, do_close=self.do_close)


# Run unit tests if called as a script
if __name__ == '__main__':
    TestSchoolStaff.do_close = False
    unittest.main()
