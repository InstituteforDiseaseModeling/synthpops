"""
Test generation of a synthetic population with microstructure.
"""

import synthpops as sp


if __name__ == '__main__':

    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 10e3
    n = int(n)
    verbose = False
    plot = False
    write = True
    return_popdict = True

    population = sp.generate_synthetic_population(n, datadir, location=location,
                                                  state_location=state_location,
                                                  country_location=country_location,
                                                  sheet_name=sheet_name,
                                                  verbose=verbose, plot=plot,
                                                  write=write, return_popdict=return_popdict)

    # for i in range(5):
    #     person = population[i]
    #     print(i, person)
