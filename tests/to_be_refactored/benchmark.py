"""Benchmark and profile SynthPops

This script produces detail profiling results for
synthpops::make_population method
it takes parameter "n" as input (default to 10001)
and produces profiling files test_{n}.txt

Example:
    $ python benchmark.py
    $ python benchmark.py 10001 20001 30001
"""
import sciris as sc
import time
import argparse
import sys
import os
from pathlib import Path
import synthpops as sp
from datetime import datetime

# to_profile = 'assign_rest_of_workers' # Must be one of the options listed below
to_profile_dict = {
    0: 'make_population',
    1: 'trim_contacts',
    2: 'generate_synthetic_population',
    3: 'generate_all_households',
    4: 'generate_larger_households',
    5: 'assign_rest_of_workers',
    6: 'make_popdict',
    7: 'make_contacts',
    8: 'sample_n_contact_ages',
    9: 'generate_living_alone',
    10: 'generate_household_head_age_by_size',
    11: 'sample_from_range'}
func_options = {
    'make_population': sp.make_population,
    'trim_contacts': sp.trim_contacts, # This is where most of the time goes for loading a population
    'generate_synthetic_population': sp.generate_synthetic_population, # This is where most of the time goes for generating a population
    'generate_all_households': sp.contact_networks.generate_all_households,
    'generate_larger_households': sp.contact_networks.generate_larger_households,
    'assign_rest_of_workers': sp.contact_networks.assign_rest_of_workers,
    'make_popdict': sp.make_popdict,
    'make_contacts': sp.make_contacts,
    'sample_n_contact_ages': sp.sample_n_contact_ages,
    'generate_living_alone': sp.contact_networks.generate_living_alone,
    'generate_household_head_age_by_size':sp.contact_networks.generate_household_head_age_by_size,
    'sample_from_range':sp.sampling.sample_from_range,
    }


def make_pop(n):
    """
    run function to create n population for input n
    """
    # n = [10000, 10001][1] # Use either a pre-generated population, or one that has to be made from scratch
    max_contacts = {'W': 10}  # only workplaces get trimmed. Schools have a separate method in the schools_module
    population = sp.make_population(n=n, max_contacts=max_contacts)
    return population


def run_benchmark(n, test_index_list, out_dir, nruns = 1, base_seed=0):
    """
    loop over list of n and output perf profile for each n to test_n.txt
    """
    test_dict = {}

    for j in range(nruns):
        for indx in test_index_list:
            print(f'running {j} {indx}')
            to_profile = to_profile_dict[indx]
            file_name = f'test_{to_profile}_{n}.txt'
            file_path = os.path.join(out_dir, file_name)
            saved_stdout = sys.stdout
            # stop is used to compute the 'average time' of execution. sc.toc() returns
            # a string of the time
            stop = 0.0
            with open(file_path, 'w') as f:
                sys.stdout = f
                start = sc.tic()
                sc.profile(run=make_pop, follow=func_options[to_profile], n=int(n))
                sc.toc()
                stop = time.time() - start
            sys.stdout.close()
            sys.stdout = saved_stdout
            if j == 0:
                test_dict[indx] = stop
            else:
                test_dict[indx] = test_dict[indx] + (stop)
    return test_dict


if __name__ == '__main__':

    default_outdir = os.path.join(os.getcwd(), 'perf_files')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='outdir', default=default_outdir, help='Output Directory')
    parser.add_argument('--runs', dest='num_runs', type=int, default=1, help='number of runs to average')
    parser.add_argument('-t', '--test', dest='test_index_list', type=int, action='append', default=[], help='test to run')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed base value')
    parser.add_argument('-t', '--test', dest='test_index_list', type=int,action='append', default=[], help='test to run')
    parser.add_argument('n', nargs='*', default=[10001], type=int, help='population')
    args = parser.parse_args()
    test_list = args.test_index_list
    if len(test_list) == 0:
        test_list = list(to_profile_dict.keys())
    nruns = args.num_runs
    runs = {}
    for n in args.n:
        print("start processing {0} , time = {1}".format(n,datetime.now().strftime("%H:%M:%S")))
        outdir = os.path.join(args.outdir, "pop_" + str(n))
        Path(outdir).mkdir(parents=True, exist_ok=True)
        runs[n] = run_benchmark(n, test_list, outdir, nruns=nruns,base_seed=args.seed)

    print("end  processing time = {0}".format(datetime.now().strftime("%H:%M:%S")))

    print("population  test                              time")

    for pop, test in runs.items():
        for t_indx, t_time in test.items():
            print("{:10d}  {:30} {:.2f}".format(pop, to_profile_dict[t_indx], t_time))
