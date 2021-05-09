"""
Test regressions using setup_regression.py
Expected files are generated automatically by running this script with regenerate = True.
This will update file in
    regression/expected/{module_name}}/
and it will be later compared with files with the same name in
     regression/report/{module_name}}/

"""

import numpy as np
import pytest
import sciris as sc
import synthpops as sp
from synthpops import data_distributions as spdd
from synthpops import base as spb
from setup_regression import regression_run, regression_validate, get_regression_dir
# Whether to regenerate files
regenerate = False
# regenerate = True


def test_regression_make_population(get_regression_dir, create_default_pop, regression_run, regression_validate):
    pop = create_default_pop
    for layer in pop.layers:
        regression_run(get_regression_dir=get_regression_dir,
                       func=get_pop_average_contacts_by_brackets,
                       params=sc.objdict(pop=pop, layer=layer),
                       filename=f"{pop.n}_seed_{pop.rand_seed}_{layer}_average_contact.json",
                       generate=regenerate)
        for method in ["density", "frequency"]:
            regression_run(get_regression_dir=get_regression_dir,
                           func=get_pop_contact_matrix,
                           params=sc.objdict(pop=pop, layer=layer, method=method),
                           filename=f"{pop.n}_seed_{pop.rand_seed}_{layer}_{method}_contact_matrix.csv",
                           generate=regenerate)

    regression_validate(get_regression_dir=get_regression_dir,
                        generate=regenerate)


def test_summary(get_regression_dir, create_default_pop, regression_run, regression_validate):
    pop = create_default_pop
    filename = "summary.json"
    regression_run(get_regression_dir=get_regression_dir,
                   func=pop.summary,
                   params=None,
                   filename=filename,
                   generate=regenerate)
    regression_validate(get_regression_dir=get_regression_dir,
                        generate=regenerate)


def get_pop_contact_matrix(pop, layer, method):
    """

    Args:
        pop (pop object)   : population, either synthpops.pop.Pop, covasim.people.People, or dict
        layer (str)        : name of the physical contact layer: H for households, S for schools, W for workplaces, C for community or other
        method: (str)      : density or frequency

    Returns:
        array: contact matrix
    """
    age_brackets = spdd.get_census_age_brackets(**pop.loc_pars)
    matrix = sp.calculate_contact_matrix(pop.popdict, method, layer)
    ageindex = spb.get_age_by_brackets(age_brackets)
    agg_matrix = spb.get_aggregate_matrix(matrix, ageindex)
    return agg_matrix


def get_pop_average_contacts_by_brackets(pop, layer, decimal=3):
    """
    get population contact counts by age brackets and save the results as json files and plots
    Args:
        pop (pop object)   : population, either synthpops.pop.Pop, covasim.people.People, or dict
        layer (str)        : name of the physical contact layer: H for households, S for schools, W for workplaces, C for community or other
        decimal (int)      : rounding precision for average contact calculation
    Returns:
    dict: a dictionary where index is the age bracket and value is the percentage of contact
    """
    # want default age brackets and to see that they line up with the average contacts by age bracket created
    age_brackets = spdd.get_census_age_brackets(**pop.loc_pars)
    average_contacts = []
    for k in age_brackets:
        degree_df = sp.count_layer_degree(pop, layer=layer, ages=age_brackets[k])
        people_in_ages = sp.filter_people(pop, ages=age_brackets[k])
        average_contacts.append(
            0 if len(degree_df) == 0 else np.round(degree_df.sum(axis=0)['degree'] / len(people_in_ages),
                                                   decimals=decimal))
    return dict(enumerate(average_contacts))


# Run pytest if called as a script
if __name__ == '__main__':
    testcase = 'test_summary'
    pytest.main(['-v', '-k', testcase])


