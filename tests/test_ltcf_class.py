import synthpops as sp
import settings
import random
import pytest
import sciris as sc
from synthpops import ltcfs
from synthpops.ltcfs import LongTermCareFacility


@pytest.fixture(scope="module")
def create_small_pop():
    return sp.Pop(n=settings.pop_sizes.small, with_facilities=1)


def test_make_ltcf(create_small_pop):
    pop = create_small_pop
    ltcf = LongTermCareFacility()
    x = random.randint(0, len(pop.ltcfs)-1)
    params = sc.objdict(ltcfid=pop.ltcfs[x]['ltcfid'],
                        resident_uids = pop.ltcfs[x]['resident_uids'],
                        staff_uids = pop.ltcfs[x]['staff_uids'])

    ltcf.set_layer_group(**params)
    for i in params:
        errmsg = f"{i} not equal!, expected: {pop.ltcfs[x][i]} actual: {ltcf[i]}"
        if hasattr(ltcf[i], "__len__"):
            assert len(ltcf[i]) == len(pop.ltcfs[x][i]), errmsg
        else:
            assert ltcf[i] == pop.ltcfs[x][i], errmsg


def test_initialize_empty_ltcfs(create_small_pop):
    pop = sc.dcp(create_small_pop)
    for n in [0, 5]:
        ltcfs.initialize_empty_ltcfs(pop, n_ltcfs=n)
        assert len(pop.ltcfs) == pop.n_ltcfs == n


def test_add_ltcf(create_small_pop):
    pop = sc.dcp(create_small_pop)
    n = len(pop.ltcfs)
    ltcf = sc.dcp(pop.ltcfs[0])
    ltcfs.add_ltcf(pop, ltcf)
    assert ltcf['ltcfid'] == pop.ltcfs[n]['ltcfid']
    assert set(ltcf['staff_uids']) == set(pop.ltcfs[n]['staff_uids'])
    assert set(ltcf['resident_uids']) == set(pop.ltcfs[n]['resident_uids'])
    assert len(pop.ltcfs) == n + 1 == pop.n_ltcfs, "after added, ltcf should be increased by 1"


def test_get_ltcf(create_small_pop):
    pop = create_small_pop
    x = random.randint(0, len(pop.ltcfs)-1)
    ltcf = ltcfs.get_ltcf(pop, x)
    assert pop.ltcfs[x]['ltcfid'] == ltcf['ltcfid']
    assert set(pop.ltcfs[x]['staff_uids']) == set(ltcf['staff_uids'])
    assert set(pop.ltcfs[x]['resident_uids']) == set(ltcf['resident_uids'])

def test_member_ages(create_small_pop):
    pop = create_small_pop
    x = random.randint(0, len(pop.ltcfs) - 1)
    ltcf = ltcfs.get_ltcf(pop, x)
    assert set(ltcf.resident_ages(pop.age_by_uid)) == \
           set([p['age'] for p in  pop.popdict.values() if p['ltcfid']==x and p['ltcf_res']]), 'Check resident_ages failed.'
    assert set(ltcf.staff_ages(pop.age_by_uid)) == \
           set([p['age'] for p in pop.popdict.values() if p['ltcfid'] == x and p['ltcf_staff']]), 'Check staff_ages failed.'
    assert set(ltcf.member_ages(pop.age_by_uid)) == \
           set([p['age'] for p in pop.popdict.values() if p['ltcfid'] == x]), 'Check member_ages failed.'

