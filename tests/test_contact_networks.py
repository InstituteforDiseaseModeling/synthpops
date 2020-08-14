"""
Test generation of a synthetic population with microstructure, reading from file, and using sp.make_population to do both.
"""

import synthpops as sp


if __name__ == '__main__':

    datadir = sp.datadir

    # location = 'seattle_metro'
    # state_location = 'Washington'
    # country_location = 'usa'
    # sheet_name = 'United States of America'
    location = 'Dakar'
    state_location = 'Dakar'
    country_location = 'Senegal'
    sheet_name = None

    n = 0.2e3
    n = int(n)
    verbose = False
    plot = False
    write = True
    return_popdict = True

    # # generate and write to file with write = True
    # population = sp.generate_synthetic_population(n, datadir, location=location,
    #                                               state_location=state_location,
    #                                               country_location=country_location,
    #                                               sheet_name=sheet_name,
    #                                               verbose=verbose, plot=plot,
    #                                               write=write, return_popdict=return_popdict)

    # # read in from file
    # population = sp.make_contacts_from_microstructure(datadir, location, state_location,
                                                      # country_location, n,)

    # # use api.py's make_population to generate on the fly
    # population = sp.make_population(n=n, generate=True)

    # use api.py's make_population to read in the population
    population = sp.make_population(n=n)

    for i in range(10):
        person = population[i]
        print(i, person)
    # sp.show_layers(population)
    # sp.show_layers(population, show_ages=True)
