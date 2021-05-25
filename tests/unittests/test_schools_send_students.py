import synthpops as sp
from synthpops import schools as sps
import random
import pytest
from collections import Counter, OrderedDict

scenarios = [
    {"name":"generate 20 students, of which 80% in pk, 20% in es this will result in not enough students to es",
     "total":20,
     "percent": {'pk': 0.8, 'es': 0.2}
     },
    {
        "name": "generate 20 students, of which 40% in pk, 60% in es, this should create enough students for both es, pk",
        "total": 20,
        "percent": {'pk': 0.4, 'es': 0.6},
    },
    {"name":"generate only 11 es students, expect to see only 1 school based on school_size_distribution_by_type",
     "total":11,
     "percent": {'pk': 0, 'es': 1}
     },
    {"name":"generate only 0 es students, expect no school generated",
     "total":0,
     "percent": {'pk': 0, 'es': 1}
     }
]


@pytest.mark.parametrize("scenarios", scenarios)
def test_send_students_to_schools(scenarios):
    # assume only 2 school brackets pk and es
    # pk is set to size 4-9 and es is size 10-15
    # pk's age (3-5) and es's age (6-12)
    # run different scenarios to make sure method returns valid list of schools
    school_size_brackets = {0: [4,5,6,7,8,9], 1:[10,11,12,13,14,15]}
    school_size_distribution_by_type = {'pk': {0:1, 1:0}, 'es': {0:0, 1:1}}
    school_type_age_ranges = {'pk': [3,4,5], 'es': [6,7,8,9,10,11,12]}

    school_types_distr_by_age = dict()
    for k,v in school_type_age_ranges.items():
        for age in v:
            school_types_distr_by_age[age] = {key: 0 for key in school_type_age_ranges.keys()}
            school_types_distr_by_age[age][k] = 1.0

    sp.log.info(scenarios["name"])
    total = scenarios["total"]
    percent = scenarios["percent"]
    run_scenario(school_type_age_ranges = school_type_age_ranges,
                 school_size_brackets = school_size_brackets ,
                 school_types_distr_by_age = school_types_distr_by_age,
                 school_size_distr_by_type = school_size_distribution_by_type,
                 total = total, percent=percent)


def random_generate_students(school_type_age_ranges, total, percentage):
    sp.logger.info(f"generating {total} random students by age brackets percentage: {percentage}")
    assert sum(percentage.values()) == 1
    student_portion = OrderedDict({key: 0 for key in percentage.keys()})
    start = 0
    for p in percentage:
        value = round(start + percentage[p]*total)
        student_portion[p] =value
        start = value

    uids_in_school = {key: 0 for key in range(0, total)}
    uids_in_school_by_age = {key: [] for key in sum(school_type_age_ranges.values(), [])}
    current_student_index = 0
    for r in student_portion:
        for i in range(current_student_index, student_portion[r]):
            uids_in_school[i] = random.randint(min(school_type_age_ranges[r]), max(school_type_age_ranges[r]))
            uids_in_school_by_age[uids_in_school[i]].append(i)
        current_student_index = student_portion[r]
    sp.logger.info(f"generating students: {uids_in_school_by_age}")
    return uids_in_school, uids_in_school_by_age


def validate_list_schools(school_type_age_ranges,
                          school_size_brackets,
                          school_size_distr_by_type,
                          student_uid_lists,
                          student_age_lists,
                          school_types,
                          allow_smallschool_size=False):
    # check no duplicate students
    students = sum(student_uid_lists, [])
    assert len(set(students)) == len(students), f"duplicate students found: {student_uid_lists}"

    # check if school's age and size matches input
    for i in range(0,len(school_types)):
        min_age = min(school_type_age_ranges[school_types[i]])
        max_age = max(school_type_age_ranges[school_types[i]])
        average_age = (sum(student_age_lists[i]) / len(student_age_lists[i]))
        assert min_age <= average_age <= max_age, \
            f"average age for {school_types[i]} must be between {min_age} and {max_age}, but we got {average_age} for {student_age_lists[i]}"
        min_size = min(school_size_brackets[[k for k, v in school_size_distr_by_type[school_types[i]].items() if v == 1][0]])
        max_size = max(school_size_brackets[[k for k, v in school_size_distr_by_type[school_types[i]].items() if v == 1][0]])
        assert len(student_uid_lists[i]) <= 2 * max_size, f"size for {school_types[i]} must not exceed twice the max_size: {max_size}\n" \
                                                      f"generated students by age: {student_age_lists[i]}"

        # normally this may happens if not enough students are available, school size may be smaller than expected.
        if not allow_smallschool_size:
            assert len(student_uid_lists[0]) >= min_size, f"size for {school_types[i]} not exceed min_size: {min_size}"


def run_scenario(school_type_age_ranges,
                 school_size_brackets,
                 school_types_distr_by_age,
                 school_size_distr_by_type,
                 total,
                 percent, allow_smallschool_size=True):
    uids_in_school, uids_in_school_by_age = random_generate_students(school_type_age_ranges, total, percent)

    sp.logger.info(f"uids_in_school:{uids_in_school}")
    sp.logger.info(f"uids_in_school_by_age: {uids_in_school_by_age}")


    # if fails you can try repro with hardcoded values captured in log above:
    # uids_in_school = {0: 3, 1: 3, 2: 4, 3: 5, 4: 4, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 11, 11: 12, 12: 10, 13: 8, 14: 12, 15: 8, 16: 10, 17: 9, 18: 8, 19: 7}
    # uids_in_school_by_age = {3: [0, 1, 5], 4: [2, 4, 6], 5: [3, 7], 6: [8], 7: [9, 19], 8: [13, 15, 18], 9: [17], 10: [12, 16], 11: [10], 12: [11, 14]}

    ages_in_school_count = {k: len(v) for k,v in uids_in_school_by_age.items()}
    sp.logger.info(f"ages_in_school_count:{ages_in_school_count}")
    sp.logger.info("call send_students_to_school_with_school_types and validate results.")
    student_age_lists, student_uid_lists, school_types = sps.send_students_to_school_with_school_types \
        (school_size_distr_by_type=school_size_distr_by_type,
         school_size_brackets=school_size_brackets,
         uids_in_school=uids_in_school,
         uids_in_school_by_age=uids_in_school_by_age,
         ages_in_school_count=ages_in_school_count,
         school_types_distr_by_age=school_types_distr_by_age,
         school_type_age_ranges=school_type_age_ranges)

    sp.log.info(f"students age lists:{student_age_lists}")
    sp.log.info(f"student_uid_lists:{student_uid_lists}")
    sp.log.info(f"school_types:{school_types}")

    validate_list_schools(school_type_age_ranges=school_type_age_ranges,
                          school_size_brackets=school_size_brackets,
                          school_size_distr_by_type=school_size_distr_by_type,
                          student_uid_lists=student_uid_lists,
                          student_age_lists=student_age_lists,
                          school_types=school_types,
                          allow_smallschool_size=allow_smallschool_size)
    if total ==0:
        assert(len(school_types) == len(student_age_lists) == len(student_uid_lists) == 0), "empty school should be generated for 0 students case"