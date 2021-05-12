"""Create zimbabwe json."""
import synthpops as sp
import pandas as pd
import os

location_data = sp.Location()
location_data.location_name = 'Zimbabwe'

new_path = os.path.join(sp.settings.datadir, 'Zimbabwe.json')
sp.save_location_to_filepath(location_data, new_path)