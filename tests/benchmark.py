# Benchmark and profile SynthPops

import sciris as sc
import synthpops as sp

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

def make_pop():
    n = [10000, 10001][1] # Use either a pre-generated population, or one that has to be made from scratch
    max_contacts = {'S': 20, 'W': 10}
    population = sp.make_population(n=n, max_contacts=max_contacts)
    return population

sc.tic()
sc.profile(run=make_pop, follow=func_options[to_profile])
sc.toc()
