#!/bin/bash

# Run this from the base synthpops folder.

python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal --state_location=Dakar
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal --state_location=Dakar --location=Dakar

python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Oregon
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Oregon --location=portland_metro

python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=seattle_metro
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Franklin_County
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Island_County
python synthpops_process_raw_data/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Spokane_County








