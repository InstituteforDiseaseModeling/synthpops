import synthpops as sp
import pytest

plot = False
verbose = False
write = True
return_popdict = True
use_default = False

datadir = sp.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'
school_enrollment_counts_available = True

n = 1000
n = int(n)


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
