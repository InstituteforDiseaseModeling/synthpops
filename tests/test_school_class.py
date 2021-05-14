import synthpops as sp
import settings
import pytest
import random
import numpy as np
import sciris as sc
from synthpops import schools as sps
from synthpops.schools import School


@pytest.fixture(scope="module")
def create_small_pop():
    return sp.Pop(n=settings.pop_sizes.small)


@pytest.fixture(scope="module")
def create_age_and_class_clustered_pop():
    return sp.Pop(n=settings.pop_sizes.small,
                  with_school_types=1,
                  school_mixing_type='age_and_class_clustered')


def test_initialize_empty_school(create_small_pop):
    pop = sc.dcp(create_small_pop)
    for n in[0, 5]:
        sps.initialize_empty_schools(pop, n_schools=n)
        assert len(pop.schools) == pop.n_schools == n

def test_(create_age_and_class_clustered_pop):
    pop = sc.dcp(create_age_and_class_clustered_pop)
    sps.initialize_empty_classrooms(pop.schools[0], 5)
    assert len(pop.schools[0]['classrooms']) == pop.schools[0]['n_classrooms'] == 5

def test_make_school(create_age_and_class_clustered_pop):
    pop = sc.dcp(create_age_and_class_clustered_pop)
    school = sp.School()
    x = random.randint(0, len(pop.schools)-1)
    print(x)
    params = sc.objdict(scid=pop.schools[x]['scid'],
                            sc_type=pop.schools[x]['sc_type'],
                            school_mixing_type=pop.schools[x]['school_mixing_type'],
                            student_uids=pop.schools[x]['student_uids'],
                            teacher_uids=pop.schools[x]['teacher_uids'],
                            non_teaching_staff_uids=pop.schools[x]['non_teaching_staff_uids'])
    school.set_layer_group(**params)
    for i in params:
        errmsg = f"{i} not equal!, expected: {pop.schools[x][i]} actual: {school[i]}"
        if hasattr(school[i], "__len__"):
            assert len(school[i]) == len(pop.schools[x][i]), errmsg
        else:
            assert school[i]==pop.schools[x][i], errmsg


def test_get_school(create_small_pop):
    # valid
    pop = create_small_pop
    scid = random.choice([s['scid'] for s in pop.schools])
    s = sps.get_school(pop, scid)
    assert set(s['student_uids']) == set([i for i in pop.popdict.keys() if pop.popdict[i]['scid']==scid and pop.popdict[i]['sc_student']])
    assert len(s) == len(set([i for i in pop.popdict.keys() if pop.popdict[i]['scid']==scid]))

    #Invalid
    max_scid = max([s['scid'] for s in pop.schools])
    with pytest.raises(IndexError):
        sps.get_school(pop, max_scid+1)

    bad_scid = "scid1"
    with pytest.raises(TypeError):
        sps.get_school(pop, bad_scid)


def test_get_class(create_age_and_class_clustered_pop):
    pop = create_age_and_class_clustered_pop
    scid = random.choice([s['scid'] for s in pop.schools])
    clid = pop.schools[scid]['classrooms'][0]['clid']
    s = pop.get_classroom(scid, clid)['student_uids']

    # check if all contacts from classroom objects match
    contacts = [v['contacts']['S'] for k,v in pop.popdict.items() if k in s and v['sc_student']]
    assert set(s) == set(np.array(contacts).flatten())


def test_add_school(create_small_pop):
    pop = sc.dcp(create_small_pop)
    scid = max([s['scid'] for s in pop.schools])
    s = sc.dcp(pop.schools[scid])
    s['scid'] = scid+1
    sps.add_school(pop, s)
    # check if adding school works
    assert max([s['scid'] for s in pop.schools]) == scid+1
    assert set(pop.schools[scid]['student_uids']) == set(pop.schools[scid+1]['student_uids'])

