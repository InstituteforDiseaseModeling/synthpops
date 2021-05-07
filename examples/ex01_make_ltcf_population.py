"""
Make a population using synthpops with long term care facilities.
"""
import synthpops as sp

pars = dict(
    n                = 10e3,
    rand_seed        = 123,
    smooth_ages      = 1,
    household_method = 'fixed_ages',
    with_facilities  = 1,
)

pop = sp.Pop(**pars)  # generate networked population
popdict = pop.to_dict()  # get dictionary version
ltcf_residents = set()  # lets find all the ltcf residents

for i, person in popdict.items():
    if person['snf_res']:
        ltcf_residents.add(i)

print(f"There are {len(ltcf_residents)} long term care facility residents in the population.")

