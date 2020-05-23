# Benchmark and profile Synthpops

import sciris as sc
import synthpops as sp

to_profile = 'generate_synthetic_population' # Must be one of the options listed below

func_options = {
    'make_population': sp.make_population,
    'trim_contacts': sp.trim_contacts, # This is where most of the time goes for loading a population
    'generate_synthetic_population': sp.generate_synthetic_population, # This is where most of the time goes for generating a population
    'get_contact_matrix_dic': sp.get_contact_matrix_dic, # ...and most of that time goes here
    'get_contact_matrix': sp.get_contact_matrix, # ...and all of that time goes here
    'make_popdict': sp.make_popdict,
    'make_contacts': sp.make_contacts,
    'sample_n_contact_ages': sp.sample_n_contact_ages,
    }

def make_pop():
    n = [20000, 5001][1] # Use either a pre-generated population, or one that has to be made from scratch
    max_contacts = {'S': 20, 'W': 10}
    population = sp.make_population(n=n, max_contacts=max_contacts)
    return population

sc.profile(run=make_pop, follow=func_options[to_profile])


