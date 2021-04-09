"""Example showing how to use the synthpops data api to load a location, modify it, and then save it."""
import synthpops as sp
import os

if __name__ == '__main__':
    # We'll load location data from here.
    input_location_filepath = "usa.json"
    # After we modify some of the location data, we'll save it here.
    output_location_filepath = "example_location.json"

    # print(f'Loading location from [{input_location_filepath}], relative to synthpops config datadir: [{synthpops.config.datadir}]')
    print(f'Loading location from [{input_location_filepath}], relative to synthpops config datadir: [{sp.datadir}]')


    # Load the location data file.  When we invoke load_location_from_filepath() below, the argument will be
    # interpreted relative to the directory provided to synthpops.config.set_datadir(). In this case,
    # we are setting that to be the python working directory.  So, the argument to load_location_from_filepath()
    # will be interpreted relative to the python working directory.

    location_data: sp.Location = sp.load_location_from_filepath(input_location_filepath)


    print('Modifying the location data...')
    # Add a note to the notes field.
    location_data.notes.append("Here's a new note added by the example code.")

    # Overwrite a field.  Here we have a community where kids are expected to do a lot of chores.
    location_data.employment_rates_by_age = [
        [1, 0.00],
        [2, 0.00],
        [3, 0.00],
        [4, 0.00],
        [5, 0.15],
        [6, 0.50]
    ]

    # Clear a field, irrespective of whatever it was set to before.
    location_data.household_size_distribution = []

    print('... done.')

    # Save the location data.
    print(f'Saving location data to [{output_location_filepath}], relative to python working directory: [{os.getcwd()}]')
    sp.save_location_to_filepath(location_data, output_location_filepath)