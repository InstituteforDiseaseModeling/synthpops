"""Example showing how to use the synthpops data api to load a location, modify it, and then save it."""
import synthpops.data
import synthpops.config
import argparse

# argparse is part of the python standard library and is the recommended way to parse command line arguments.
# For more information, see this link: https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser()

parser.add_argument("--input_location_filepath",
                    type=str,
                    default=None,
                    required=True,
                    help='Input location file path, relative to current directory.')

parser.add_argument("--output_filepath",
                    type=str,
                    default=None,
                    required=True,
                    help="Output file path.")

args = parser.parse_args()

if __name__ == '__main__':
    print(f'Loading location from [{args.input_location_filepath}]')

    # Load the location data file.  When we invoke load_location_from_filepath() below, the argument will be
    # interpreted relative to the directory provided to synthpops.config.set_datadir(). In this case,
    # we are setting that to be the python working directory.  So, the argument to load_location_from_filepath()
    # will be interpreted relative to the python working directory.
    synthpops.config.set_datadir(".")
    location_data: synthpops.data.Location = synthpops.data.load_location_from_filepath(args.input_location_filepath)

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
    print(f'Saving location data to [{args.output_filepath}]')
    synthpops.data.save_location_to_filepath(location_data, args.output_filepath)