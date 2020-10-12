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
datadir = str(pathlib.Path(testdir, "../data").absolute())
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
age_brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
age_brackets_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in sorted(age_brackets.keys())]


for seed in range(1, 100, 100):
    test_prefix = f"{n}_seed{seed}"
    print("seed:", seed)  # Random seed
    sc.tic()
    pop = sp.make_population(n=n, generate=True, rand_seed=seed)
    sc.toc()

    # for code in ['H', 'S', 'W']:
    #     average = utilities.get_average_contact_by_age(pop,
    #                                                    datadir=datadir,
    #                                                    state_location=state_location,
    #                                                    country_location=country_location, code=code, decimal=3)
    #     utilities.plot_array(average, datadir=figdir, testprefix=f"{code} {test_prefix} contact by age", expect_label="contacts",
    #                          xlabels=age_brackets_labels, xlabel_rotation=50)

    # utilities.check_age_distribution(pop, n,
    #                                  datadir=datadir, figdir=figdir,
    #                                  location=location, state_location=state_location, country_location=country_location,
    #                                  test_prefix=test_prefix,
    #                                  skip_stat_check=True)

    # utilities.check_enrollment_distribution(pop, n,
    #                                         datadir=datadir, figdir=figdir,
    #                                         location=location, state_location=state_location, country_location=country_location,
    #                                         test_prefix=test_prefix,
    #                                         skip_stat_check=True, plot_only=True)

    # utilities_dist.check_work_size_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
    #                                     state_location=state_location,
    #                                     test_prefix=test_prefix,
    #                                     country_location=country_location)

    # utilities_dist.check_employment_age_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
    #                                          state_location=state_location,
    #                                          country_location=country_location,
    #                                          test_prefix=test_prefix,
    #                                          use_default=True)

    # utilities_dist.check_household_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
    #                                     state_location=state_location,
    #                                     country_location=country_location,
    #                                     test_prefix=test_prefix,
    #                                     use_default=True)

    # utilities_dist.check_school_size_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
    #                                       location="seattle_metro",
    #                                       state_location=state_location,
    #                                       country_location=country_location,
    #                                       test_prefix=test_prefix,
    #                                       use_default=True)

    utilities_dist.check_household_head(pop=pop, n=n, datadir=datadir,
                                        figdir=figdir,
                                        state_location=state_location,
                                        country_location=country_location,
                                        test_prefix=test_prefix,
                                        use_default=True)