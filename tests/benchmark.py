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
import synthpops as sp
import argparse
import sys

to_profile = 'assign_rest_of_workers' # Must be one of the options listed below

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
    }

def make_pop(n):
    """
        run function to create n population for input n
    """
    #n = [10000, 10001][1] # Use either a pre-generated population, or one that has to be made from scratch
    max_contacts = {'S': 20, 'W': 10}
    population = sp.make_population(n=n, max_contacts=max_contacts)
    return population

def run_benchmark(list_of_n):
    """
    loop over list of n and output perf profile for each n to test_n.txt
    """
    for n in list_of_n:
        saved_stdout = sys.stdout
        with open(f'test_{n}.txt', 'w') as f:
            sys.stdout = f
            sc.tic()
            sc.profile(run=make_pop, follow=func_options[to_profile], n=int(n))
            sc.toc()
        sys.stdout.close()
        sys.stdout = saved_stdout
        print(f'result n={n} : test_{n}.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', nargs='*', default=[10001], help='population')
    args = parser.parse_args()
    run_benchmark(args.n)
