#!/bin/bash

# Run this from the base synthpops folder.

python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal --state_location=Dakar
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=Senegal --state_location=Dakar --location=Dakar

python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Oregon
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Oregon --location=portland_metro

python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=seattle_metro
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Franklin_County
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Island_County
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=Spokane_County








