import synthpops as sp

"""
An of how to make synthetic populations with microstructure (households, schools, and workplaces)
Populations have demographics (age, sex) from data.
Not an exhaustive list of what synthpops can do - please take a look through the code base for the many possibilities.
"""

if __name__ == '__main__':

    datadir = sp.datadir # point datadir where your data folder lives

    # location information - currently we only support the Seattle Metro area in full, however other locations can be supported with this framework at a later date
    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 5000
    n = int(n)
    verbose = False
    plot = False
    write = False
    return_popdict = True

    # this will generate a population with microstructure and age demographics that approximate those of the location selected
    # also saves to file in:
    #    datadir/demographics/contact_matrices_152_countries/state_location/
    population = sp.generate_synthetic_population(n, datadir, location=location, state_location=state_location, country_location=country_location, sheet_name=sheet_name, verbose=verbose, plot=plot, write=write, return_popdict=return_popdict)

    # load that population into a dictionary of individuals who know who their contacts are (loading in from written files)
    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': n}
    contacts = sp.make_contacts(location=location, state_location=state_location, country_location=country_location, options_args=options_args, network_distr_args=network_distr_args)

    # verbose = True
    verbose = False
    show_ages = True
    # show_ages = False
    # show the contact ages by layer for a few people (10)
    if verbose:
        sp.show_layers(contacts,show_ages=True, show_n=10)
