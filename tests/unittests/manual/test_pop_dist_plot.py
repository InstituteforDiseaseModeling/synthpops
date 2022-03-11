"""
This test will produce plots of below distributions for the generated population
in comparison with the expected data
note that tests looping over seeds 1 to 5000 with increment of 100, please change line 32 if needed
results will be saved to dist_reports folder
* age distribution
* average contacts by age
* enrollment by age
* workplace size
* employment by age
* school size
* household size
* household size vs household head age
"""
import os
import pathlib
import synthpops as sp
import sciris as sc
import sys
import shutil

testdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = sp.datadir
figdir = os.path.join(os.path.dirname(__file__), "dist_reports")
shutil.rmtree(figdir, ignore_errors=True)
os.makedirs(figdir, exist_ok=True)

sys.path.append(testdir)
import utilities_dist
import utilities

n = 20001
location = "seattle_metro"
state_location = "Washington"
country_location = "usa"

age_brackets = sp.get_census_age_brackets(datadir=datadir,
                                          state_location=state_location,
                                          country_location=country_location)
age_brackets_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in sorted(age_brackets.keys())]
with_school_types = True
school_mixing_type = 'random'
school_type = None
if with_school_types:
    # use synthpops to find the school types available for the location
    expected_school_size_distr = sp.get_school_size_distr_by_type(sp.datadir, location=location, state_location=state_location, country_location=country_location)
    school_type = sorted(expected_school_size_distr.keys())

for seed in range(1, 100, 100):
    test_prefix = f"{n}_seed{seed}"
    print("seed:", seed)  # Random seed
    params = dict(n=n,
                 location=location,
                 state_location=state_location,
                 country_location=country_location,
                 generate=True,
                 rand_seed=seed,
                 with_school_types=with_school_types,
                 school_mixing_type=school_mixing_type)
    sc.tic()
    pop = sp.make_population(**params)
    sc.toc()

    for setting_code in ['H', 'S', 'W']:
        average = utilities.get_average_contact_by_age(pop,
                                                       datadir=datadir,
                                                       state_location=state_location,
                                                       country_location=country_location, setting_code=setting_code, decimal=3)
        utilities.plot_array(average, datadir=figdir, testprefix=f"{setting_code} {test_prefix} contact by age bracket", expect_label="contacts",
                             names=age_brackets_labels, xlabel_rotation=50)

    utilities_dist.check_age_distribution(pop, n,
                                          datadir=datadir, figdir=figdir,
                                          location=location, state_location=state_location, country_location=country_location,
                                          test_prefix=test_prefix,
                                          skip_stat_check=True)

    utilities_dist.check_enrollment_distribution(pop, n,
                                                 datadir=datadir, figdir=figdir,
                                                 location=location, state_location=state_location, country_location=country_location,
                                                 test_prefix=test_prefix,
                                                 skip_stat_check=True, plot_only=True, school_type=school_type)

    utilities_dist.check_work_size_distribution(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                                state_location=state_location,
                                                test_prefix=test_prefix,
                                                country_location=country_location)

    utilities_dist.check_employment_age_distribution(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                                     state_location=state_location,
                                                     country_location=country_location,
                                                     test_prefix=test_prefix,
                                                     use_default=True)

    utilities_dist.check_household_distribution(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                                state_location=state_location,
                                                country_location=country_location,
                                                test_prefix=test_prefix,
                                                use_default=True)

    utilities_dist.check_school_size_distribution(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                                  location=location,
                                                  state_location=state_location,
                                                  country_location=country_location,
                                                  test_prefix=test_prefix,
                                                  use_default=True,
                                                  school_type=school_type)

    utilities_dist.check_household_head(pop=pop, n=n, datadir=datadir,
                                        figdir=figdir,
                                        state_location=state_location,
                                        country_location=country_location,
                                        test_prefix=test_prefix,
                                        use_default=True)