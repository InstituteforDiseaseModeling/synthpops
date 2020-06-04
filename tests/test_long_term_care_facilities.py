import sciris as sc
import synthpops as sp

do_save = True
do_plot = False
verbose = False
# write = True
write = False
return_popdict = True
use_default = False

datadir = sp.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'
school_enrollment_counts_available = True

gen_pop_size = 60e3
gen_pop_size = int(gen_pop_size)

# # generate and write to file
# popdict = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location, gen_pop_size, school_enrollment_counts_available=school_enrollment_counts_available, write=write, do_plot=do_plot, return_popdict=return_popdict)

# # read in from file
# popdict = sp.make_contacts_with_facilities_from_microstructure(datadir, location, state_location, country_location, gen_pop_size)

# # generate on the fly
popdict = sp.make_population(n=gen_pop_size, generate=True, with_facilities=True)


# sp.check_all_residents_are_connected_to_staff(popdict)
