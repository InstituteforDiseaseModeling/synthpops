import sciris as sc
import synthpops as sp
import pytest

plot = False
verbose = False
# write = True
write = False
return_popdict = True
use_default = False

datadir = sp.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'

use_two_group_reduction = True
average_LTCF_degree = 20
ltcf_staff_age_min = 20
ltcf_staff_age_max = 60


with_school_types = True
average_class_size = 20
inter_grade_mixing = 0.1
average_student_teacher_ratio = 20
average_teacher_teacher_degree = 3
teacher_age_min = 25
teacher_age_max = 75

with_non_teaching_staff = True
# if with_non_teaching_staff is False, but generate is True, then average_all_staff_ratio should be average_student_teacher_ratio or 0
average_student_all_staff_ratio = 11
average_additional_staff_degree = 20
staff_age_min = 20
staff_age_max = 75

# school_mixing_type = 'random'
# school_mixing_type = 'clustered'
school_mixing_type = {'pk': 'clustered', 'es': 'random', 'ms': 'clustered', 'hs': 'random', 'uv': 'random'}


rand_seed = 1


n = 1000
n = int(n)


def test_generate_microstructures_with_non_teaching_staff():
    # # generate and write to file
    population1 = sp.generate_microstructure_with_facilities(datadir,
                                                            location=location,
                                                            state_location=state_location,
                                                            country_location=country_location,
                                                            gen_pop_size=n,
                                                            use_two_group_reduction=use_two_group_reduction,
                                                            average_LTCF_degree=average_LTCF_degree,
                                                            ltcf_staff_age_min=ltcf_staff_age_min,
                                                            ltcf_staff_age_max=ltcf_staff_age_max,
                                                            with_school_types=with_school_types,
                                                            school_mixing_type=school_mixing_type,
                                                            average_class_size=average_class_size,
                                                            inter_grade_mixing=inter_grade_mixing,
                                                            average_student_teacher_ratio=average_student_teacher_ratio,
                                                            average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                            teacher_age_min=teacher_age_min,
                                                            teacher_age_max=teacher_age_max,
                                                            average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                            average_additional_staff_degree=average_additional_staff_degree,
                                                            staff_age_min=staff_age_min,
                                                            staff_age_max=staff_age_max,
                                                            write=write,
                                                            plot=plot,
                                                            return_popdict=return_popdict,
                                                            use_default=use_default)

    # # # read in from file
    population2 = sp.make_contacts_with_facilities_from_microstructure(datadir,
                                                                      location=location,
                                                                      state_location=state_location,
                                                                      country_location=country_location,
                                                                      n=n,
                                                                      use_two_group_reduction=use_two_group_reduction,
                                                                      average_LTCF_degree=average_LTCF_degree,
                                                                      with_school_types=with_school_types,
                                                                      school_mixing_type=school_mixing_type,
                                                                      average_class_size=average_class_size,
                                                                      inter_grade_mixing=inter_grade_mixing,
                                                                      average_student_teacher_ratio=average_student_teacher_ratio,
                                                                      average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                                      average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                                      average_additional_staff_degree=average_additional_staff_degree)


    # # generate on the fly
    sc.tic()
    population3 = sp.make_population(n=n,
                                    generate=True,
                                    with_facilities=True,
                                    use_two_group_reduction=use_two_group_reduction,
                                    average_LTCF_degree=average_LTCF_degree,
                                    ltcf_staff_age_min=ltcf_staff_age_min,
                                    ltcf_staff_age_max=ltcf_staff_age_max,
                                    with_school_types=with_school_types,
                                    school_mixing_type=school_mixing_type,
                                    average_class_size=average_class_size,
                                    inter_grade_mixing=inter_grade_mixing,
                                    average_student_teacher_ratio=average_student_teacher_ratio,
                                    average_teacher_teacher_degree=average_teacher_teacher_degree,
                                    teacher_age_min=teacher_age_min,
                                    teacher_age_max=teacher_age_max,
                                    with_non_teaching_staff=with_non_teaching_staff,
                                    average_student_all_staff_ratio=average_student_all_staff_ratio,
                                    average_additional_staff_degree=average_additional_staff_degree,
                                    staff_age_min=staff_age_min,
                                    staff_age_max=staff_age_max,
                                    rand_seed=rand_seed)
    sc.toc()

    sp.check_all_residents_are_connected_to_staff(population3)

    return population1, population2, population3


# generate and write to file
def test_generate_microstructures_with_facilities():
    popdict = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location, n,
                                                         write=write, plot=False, return_popdict=return_popdict)
    assert (len(popdict) is not None)
    return popdict


# read in from file
def test_make_contacts_with_facilities_from_microstructure():
    popdict = sp.make_contacts_with_facilities_from_microstructure(datadir, location, state_location, country_location,
                                                                   n)

    # verify these keys are either None or set to a value and LTCF contacts exist
    for i, uid in enumerate(popdict):
        print(popdict[uid]['hhid'])
        if popdict[uid]['hhid'] is None:
            assert popdict[uid]['hhid'] is None
        else:
            assert popdict[uid]['hhid'] is not None
        if popdict[uid]['scid'] is None:
            assert popdict[uid]['scid'] is None
        else:
            assert popdict[uid]['scid'] is not None
        if popdict[uid]['wpid'] is None:
            assert popdict[uid]['wpid'] is None
        else:
            assert popdict[uid]['wpid'] is not None
        assert popdict[uid]['contacts']['LTCF'] is not None
    return popdict


# generate on the fly
def test_make_population():
    population = sp.make_population(n=n, generate=True, with_facilities=True, use_two_group_reduction=True,
                                    average_LTCF_degree=20)

    for key, person in population.items():
        assert population[key]['contacts'] is not None
        assert population[key]['contacts']['LTCF'] is not None
        assert len(population[key]['contacts']['H']) >= 0

    expected_layers = {'H', 'S', 'W', 'C', 'LTCF'}

    for layerkey in population[key]['contacts'].keys():
        if layerkey in expected_layers:
            assert True
        else:
            assert False

    return population


def test_make_population_with_industry_code():
    popdict = sp.make_population(n=n, generate=True, with_industry_code=True, use_two_group_reduction=True,
                                 average_LTCF_degree=20)

    # verify these keys are either None or set to a value
    for i, uid in enumerate(popdict):
        if popdict[uid]['wpid'] is None:
            assert popdict[uid]['wpid'] is None
        else:
            assert popdict[uid]['wpid'] is not None
        if popdict[uid]['wpindcode'] is None:
            assert popdict[uid]['wpindcode'] is None
        else:
            assert popdict[uid]['wpindcode'] is not None

    return popdict


def test_make_population_with_multi_flags():
    with pytest.raises(ValueError, match=r"Requesting both long term*") as info:
        sp.make_population(n=n, generate=True, with_industry_code=True, with_facilities=True)
    return info


if __name__ == '__main__':

    generate_micro_popdict = test_generate_microstructures_with_facilities()

    make_contacts_popdict = test_make_contacts_with_facilities_from_microstructure()

    make_pop_popdict = test_make_population()

    make_popdict_with_industry = test_make_population_with_industry_code()

    make_pop_with_facilities_industry_fails = test_make_population_with_multi_flags()

    population = sp.make_population(n=n,
                                    generate=True,
                                    with_facilities=True,
                                    use_two_group_reduction=use_two_group_reduction,
                                    average_LTCF_degree=average_LTCF_degree,
                                    ltcf_staff_age_min=ltcf_staff_age_min,
                                    ltcf_staff_age_max=ltcf_staff_age_max,
                                    with_school_types=with_school_types,
                                    school_mixing_type=school_mixing_type,
                                    average_class_size=average_class_size,
                                    inter_grade_mixing=inter_grade_mixing,
                                    average_student_teacher_ratio=average_student_teacher_ratio,
                                    average_teacher_teacher_degree=average_teacher_teacher_degree,
                                    teacher_age_min=teacher_age_min,
                                    teacher_age_max=teacher_age_max,
                                    with_non_teaching_staff=with_non_teaching_staff,
                                    average_student_all_staff_ratio=average_student_all_staff_ratio,
                                    average_additional_staff_degree=average_additional_staff_degree,
                                    staff_age_min=staff_age_min,
                                    staff_age_max=staff_age_max,
                                    rand_seed=rand_seed)

    sp.check_all_residents_are_connected_to_staff(population)
