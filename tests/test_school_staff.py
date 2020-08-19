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
import sciris as sc
import inspect

#%% Utility functions for running tests

def runpop(resultdir, actual_vals, testprefix, method):

    """
    run any method which create apopulation
    and write args and population to file "{resultdir}/{testprefix}.txt"
    method must be a method which returns population
    and write population file to "{resultdir}/{testprefix}_pop.json"

    args:
      resultdir (str): result folder
      actual_vals (dict): a dictionary with param name and param value
      testprefix (str): test prefix to generate file name
    """
    os.makedirs(resultdir, exist_ok=True)
    params = {}
    args = inspect.getfullargspec(method).args
    for i in range(0, len(args)):
        params[args[i]] = inspect.signature(method).parameters[args[i]].default
    for name in actual_vals:
        if name in params.keys():
            params[name] = actual_vals[name]
    with open(os.path.join(resultdir, f"{testprefix}.txt"), mode="w") as cf:
        for key, value in params.items():
            cf.writelines(str(key) + ':' + str(value) + "\n")

    pop = method(**params)
    sc.savejson(os.path.join(resultdir, f"{testprefix}_pop.json"), pop, indent=2)
    return pop


def copy_input(sourcedir, resultdir, subdir_level):

    """
    Copy files to the target datadir up to the subdir level
    """

    # copy all files to datadir except the ignored files
    ignorepatterns = shutil.ignore_patterns("*contact_networks*",
                                            "*contact_networks_facilities*",
                                            "*New_York*",
                                            "*Oregon*")
    shutil.copytree(sourcedir, os.path.join(resultdir, subdir_level), ignore=ignorepatterns)


def check_teacher_staff_ratio(pop, average_student_teacher_ratio, average_student_all_staff_ratio, err_margin=0):

    """
    check if generated population matches
    average_student_teacher_ratio and average_student_all_staff_ratio

    """
    i = 0
    school = {}
    for p in pop.values():
        if p["scid"] is not None:
            row = {"scid": p["scid"],
                   "student": 0 if p["sc_student"] is None else p["sc_student"],
                   "teacher": 0 if p["sc_teacher"] is None else p["sc_teacher"],
                   "staff": 0 if p["sc_staff"] is None else p["sc_staff"]}
            school[i] = row
            i += 1
    df_school = pd.DataFrame.from_dict(school).transpose()
    result = df_school.groupby('scid', as_index=False)[['student', 'teacher', 'staff']].agg(lambda x: sum(x))

    print(result.head(20))

    # check for 0 staff/teacher case to see if it is dues to school size being too small
    zero_teacher_case = result.query('teacher == 0 & student > @average_student_teacher_ratio')
    assert(len(zero_teacher_case) == 0), \
        f"All schools with more students than the student teacher ratio should have at least one teacher. {len(zero_teacher_case)} did not."
    zero_staff_case = result.query('staff == 0 & student > @average_student_all_staff_ratio')
    assert(len(zero_staff_case) == 0), \
        f"All schools with more students than the student staff ratio: {average_student_all_staff_ratio} should have at least 1 staff. {len(zero_staff_case)} did not."

    # exclude 0 teacher if size is too small
    result = result[result.teacher > 0][result.staff > 0]
    result["teacher_ratio"] = result["student"] / (result["teacher"])
    result["allstaff_ratio"] = result["student"] / (result["teacher"] + result["staff"])

    # average across school must match input
    actual_teacher_ratio = np.round(result["teacher_ratio"].mean())
    assert (int(average_student_teacher_ratio + err_margin) >= actual_teacher_ratio >= int(average_student_teacher_ratio - err_margin)), \
        f"teacher ratio: expected: {average_student_teacher_ratio} actual: {actual_teacher_ratio}"
    actual_staff_ratio = np.round(result["allstaff_ratio"].mean())
    assert (int(average_student_all_staff_ratio + err_margin) >= actual_staff_ratio >= int(average_student_all_staff_ratio - err_margin)), \
        f"all staff ratio expected: {average_student_all_staff_ratio} actual: {actual_staff_ratio}"
    return result



#%% Actual tests

class TestSchoolStaff(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resultdir = tempfile.TemporaryDirectory().name
        cls.dataDir = os.path.join(cls.resultdir, "data")
        cls.subdir_level = "data/demographics/contact_matrices_152_countries"
        cls.sourcedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls.subdir_level)
        copy_input(cls.sourcedir, cls.resultdir, cls.subdir_level)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.resultdir, ignore_errors=True)

    @classmethod
    def copy_input(cls):

        # copy all files to datadir except the ignored files
        ignorepatterns = shutil.ignore_patterns("*contact_networks*",
                                                "*contact_networks_facilities*",
                                                "*New_York*",
                                                "*Oregon*")
        shutil.copytree(cls.sourcedir, os.path.join(cls.resultdir, cls.subdir_level), ignore=ignorepatterns)

    def test_staff_generate(self):

        """
        generate 10001 population and check if teacher/staff ratio match
        """
        seed = 1
        sp.set_seed(seed)
        # set param
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
        pop = runpop(resultdir=self.resultdir, testprefix="staff_generate", actual_vals=vals, method=sp.generate_synthetic_population)
        result = check_teacher_staff_ratio(pop, average_student_teacher_ratio, average_student_all_staff_ratio)

    def test_with_ltcf(self):
        """
        test with long term care facilities options
        """
        seed = 1
        sp.set_seed(seed)
        # set param
        n = 10001
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
        pop = runpop(resultdir=self.resultdir, testprefix="staff_ltcf", actual_vals=vals,
                                   method=sp.generate_microstructure_with_facilities)
        result = check_teacher_staff_ratio(pop, average_student_teacher_ratio,
                                                         average_student_all_staff_ratio, err_margin=2)
