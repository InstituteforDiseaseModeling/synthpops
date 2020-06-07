import synthpops as sp

plot = False
verbose = False
write = True
# write = False
return_popdict = True
use_default = False

datadir = sp.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'
school_enrollment_counts_available = True

# n = 10e3
n = 200
n = int(n)

# # generate and write to file when True
# population = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location, n, write=write, plot=plot, return_popdict=return_popdict)

# # read in from file
# population = sp.make_contacts_with_facilities_from_microstructure(datadir, location, state_location, country_location, n)

# # use api.py's make_population to generate on the fly
# population = sp.make_population(n=n, generate=True, with_facilities=True)

# # use api.py's make_population to read in the population
population = sp.make_population(n=n, with_facilities=True)


sp.check_all_residents_are_connected_to_staff(population)
