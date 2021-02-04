#!/bin/bash

# Run this from the base synthpops folder.

python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington
python scripts/migrate_legacy_data.py --output_folder=data/ --country_location=usa --state_location=Washington --location=seattle_metro

