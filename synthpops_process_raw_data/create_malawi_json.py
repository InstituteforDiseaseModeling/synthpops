"""Create malawi json."""
import synthpops as sp
import pandas as pd
import os

filepath = os.path.join(sp.settings.datadir, 'Malawi.json')
raw_data_path = os.path.join(sp.settings.datadir, 'Malawi')

location_data: sp.Location = sp.load_location_from_filepath(filepath)

employment_rates_df = pd.read_csv(os.path.join(raw_data_path, 'Malawi_employment_rates_by_age.csv'))
employment_rates_arr = sp.convert_df_to_json_array(employment_rates_df, cols=employment_rates_df.columns, int_cols=['age'])

location_data.employment_rates_by_age = employment_rates_arr

new_path = os.path.join(sp.settings.datadir, 'Malawi.json')
sp.save_location_to_filepath(location_data, new_path)