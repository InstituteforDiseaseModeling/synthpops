"""
test school staff features
"""
import unittest
import tempfile
import os
import sys
import shutil
import pandas as pd
import numpy as np
import synthpops as sp
import testutilities

class TestSchoolStaff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resultdir = tempfile.TemporaryDirectory().name
        cls.dataDir = os.path.join(cls.resultdir, "data")
        cls.subdir_level = "data/demographics/contact_matrices_152_countries"
        cls.sourcedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls.subdir_level)
        testutilities.copy_input(cls.sourcedir, cls.resultdir, cls.subdir_level)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.resultdir, ignore_errors=True)

    @classmethod
    def copy_input(cls):
        to_exclude = [os.path.join(cls.sourcedir, "contact_networks"),
                      os.path.join(cls.sourcedir, "contact_networks_facilities")]


        # copy all files to datadir except the ignored files
        ignorepatterns = shutil.ignore_patterns("*contact_networks*",
                                                "*contact_networks_facilities*",
                                                "*New_York*",
                                                "*Oregon*")
        shutil.copytree(cls.sourcedir, os.path.join(cls.resultdir, cls.subdir_level), ignore=ignorepatterns)

    def testStaffGenerate(self):

        """
        generate 10001 population and check if teacher/staff ratio match
        """
        seed = 1
        sp.set_seed(seed)
        #set param
        n = 10001
        datadir = self.dataDir
        location = 'seattle_metro'
        state_location = 'Washington'
        country_location = 'usa'
        sheet_name = 'United States of America'
        school_enrollment_counts_available = False
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
        pop = testutilities.runpop(resultdir=self.resultdir, testprefix="staff_generate", actual_vals=vals, method=sp.generate_synthetic_population)
        result = testutilities.check_teacher_staff_ratio(pop, average_student_teacher_ratio, average_student_all_staff_ratio)

    def testWithlTCF(self):
        seed = 1
        sp.set_seed(seed)
        # set param
        gen_pop_size = 10001
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
        average_student_teacher_ratio = 20
        average_teacher_teacher_degree = 3
        teacher_age_min = 25
        teacher_age_max = 70

        with_non_teaching_staff = True
        average_student_all_staff_ratio = 11
        average_additional_staff_degree = 20
        staff_age_min = 20
        staff_age_max = 75
        school_mixing_type = {'pk': 'clustered', 'es': 'random', 'ms': 'clustered', 'hs': 'random', 'uv': 'random'}
        return_popdict = True

        vals = locals()
        pop = testutilities.runpop(resultdir=self.resultdir, testprefix="staff_ltcf", actual_vals=vals,
                                   method=sp.generate_microstructure_with_facilities)
        result = testutilities.check_teacher_staff_ratio(pop, average_student_teacher_ratio,
                                                         average_student_all_staff_ratio, err_margin=1)