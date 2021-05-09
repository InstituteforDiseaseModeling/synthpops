"""
Tests for school releated parameters are set correctly
* teacher_age_min
* teacher_age_max
* staff_age_min
* staff_age_max
"""
import synthpops as sp
import pytest

pars = dict(
    n=5e3,
    rand_seed=1,
    max_contacts=None,
    country_location='usa',
    state_location='Washington',
    location='seattle_metro',
    use_default=True,
    with_non_teaching_staff=1,
    with_school_types=1,
    school_mixing_type={'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},
    average_class_size=20,
    inter_grade_mixing=0.1
)
# test data for (teacher_age_min, teacher_age_max, staff_age_min, staff_age_max)
testdata = [
    (25, 75, 20, 75),
    (18, 80, 16, 65),
    (20, 99, 18, 60)
]
@pytest.mark.parametrize("tmin,tmax,smin,smax", testdata)
def test_age_bounds(tmin,tmax,smin,smax):
    """
    Generate population based on teacher and staff age limits, verify if actual age is within expected range
    Args:
        tmin: teacher_age_min
        tmax: teacher_age_max
        smin: staff_age_min
        smax: staff_age_min

    Returns:
        None
    """
    pars["teacher_age_min"] = tmin
    pars["teacher_age_max"] = tmax
    pars["staff_age_min"] = smin
    pars["staff_age_max"] = smax
    print(f"teacher_age_min:{tmin}, teacher_age_max:{tmax}, staff_age_min:{smin}, staff_age_max:{smax}")
    pop = sp.Pop(**pars)
    verify_age_bounds(pop.popdict,tmin,tmax,smin,smax)


def verify_age_bounds(popdict, tmin, tmax, smin, smax):
    """
    Loop over the dictionary to check if teacher and staff age is within valid range
    Args:
        popdict: popdict object of population generated
        tmin: teacher_age_min
        tmax: teacher_age_max
        smin: staff_age_min
        smax: staff_age_max

    Returns:
        None
    """
    actual_tmin = min(d['age'] for d in popdict.values() if d['sc_teacher'])
    actual_tmax = max(d['age'] for d in popdict.values() if d['sc_teacher'])
    actual_smin = min(d['age'] for d in popdict.values() if d['sc_staff'])
    actual_smax = max(d['age'] for d in popdict.values() if d['sc_staff'])
    print(f"generated teachers' age: [{actual_tmin}:{actual_tmax}]")
    print(f"generated staffs' age: [{actual_smin}:{actual_smax}]")
    assert actual_tmin >= tmin and actual_tmax <= tmax, \
        f"invalid teachers' age [{actual_tmin}:{actual_tmax}] but expected [{tmin}:{tmax}]"
    assert actual_smin >= smin and actual_smax <= smax, \
        f"invalid staffs' age [{actual_smin}:{actual_smax}] but expected [{smin}:{smax}]"




