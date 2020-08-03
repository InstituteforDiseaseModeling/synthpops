import synthpops as sp
import sciris as sc
import inspect
import os
import numpy as np
import pandas as pd
import shutil

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
    assert(len(zero_teacher_case) == 0), "some school has enough students but no teacher"
    zero_staff_case = result.query('staff == 0 & student > @average_student_all_staff_ratio')
    assert(len(zero_staff_case) == 0), "some school has enough students but no staff"

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
