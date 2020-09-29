import os
import pathlib
import synthpops as sp
import sys
import shutil
n = 20001
testdir = os.path.dirname(os.path.dirname(__file__))
datadir = str(pathlib.Path(testdir, "../data").absolute())
figdir = os.path.join(os.path.dirname(__file__), "dist_reports")
shutil.rmtree(figdir, ignore_errors=True)
os.makedirs(figdir, exist_ok=True)
state_location = "Washington"
country_location = "usa"
sys.path.append(testdir)
import utilities_dist
for seed in range(1, 5000, 100):
    test_prefix=f"{n}_seed{seed}"
    pop = sp.make_population(n=n, generate=True, rand_seed=seed)
    utilities_dist.check_work_size_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                        state_location=state_location,
                                        test_prefix=test_prefix,
                                        country_location=country_location)

    utilities_dist.check_employment_age_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                             state_location=state_location,
                                             country_location=country_location,
                                             test_prefix=test_prefix,
                                             use_default=True)

    utilities_dist.check_household_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                        state_location=state_location,
                                        country_location=country_location,
                                        test_prefix=test_prefix,
                                        use_default=True)

    utilities_dist.check_school_size_dist(pop=pop, n=n, datadir=datadir, figdir=figdir,
                                          location="seattle_metro",
                                          state_location=state_location,
                                          country_location=country_location,
                                          test_prefix=test_prefix,
                                          use_default=True)

    utilities_dist.check_household_head(pop=pop, n=n, datadir=datadir,
                                        figdir=figdir,
                                        state_location=state_location,
                                        country_location=country_location,
                                        test_prefix=test_prefix,
                                        use_default=True)

    #temprary workaround to move all the figure to report folder
    for f in os.listdir(os.path.dirname(__file__)):
        if f.endswith('png'):
            shutil.move(os.path.join(os.path.dirname(__file__), f), figdir)