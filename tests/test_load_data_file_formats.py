import os
import shutil
import pandas as pd

import synthpops as sp

sp_datadir = sp.datadir

import unittest


class DataFileFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.is_debugging = False

    def tearDown(self) -> None:
        pass

    def copy_dat_and_make_csv(self, dat_fullpath):
        """
          copies a .dat file locally with a DEBUG_prefix and makes a copy as .csv
          returns: .dat filename, .csv filename
        """
        dat_full_filename = os.path.basename(dat_fullpath)
        dat_filetitle = os.path.splitext(dat_full_filename)[0]
        dat_localfilename = f"DEBUG_{dat_filetitle}.dat"
        csv_localfilename = f"DEBUG_{dat_filetitle}.csv"
        shutil.copyfile(
            src=dat_fullpath,
            dst=dat_localfilename
        )
        tmp_dataframe = pd.read_csv(dat_localfilename)
        tmp_dataframe.to_csv(csv_localfilename)
        return dat_localfilename, csv_localfilename

    def test_csv_loads_same_as_dat(self):

        headage_householdsize_distribution_path = sp.get_household_head_age_by_size_path(
            datadir=sp_datadir,
            state_location="Washington",
            country_location="usa"
        )

        headage_distro_datfilename, headage_distro_csvfilename = self.copy_dat_and_make_csv(
            headage_householdsize_distribution_path)

        headage_distro_dat_df = sp.get_household_head_age_by_size_df(datadir="", file_path=headage_distro_datfilename)
        headage_distro_csv_df = sp.get_household_head_age_by_size_df(datadir="", file_path=headage_distro_csvfilename)
        if self.is_debugging:
            print(headage_distro_dat_df.describe())
            print(headage_distro_dat_df.columns)
        dat_household_age_18_20_weights = headage_distro_dat_df['household_head_age_18_20']
        dat_household_size_4_reference_35_39 = headage_distro_dat_df['household_head_age_35_39'][5]
        dat_household_size_1_weights = headage_distro_dat_df.loc[0]

        if self.is_debugging:
            print("Household age 18_20 weights")
            print(dat_household_age_18_20_weights)

            print("Household size 1 relative weights")
            print(dat_household_size_1_weights)

            print("Household size 4, reference person 25-39 relative weight")
            print(dat_household_size_4_reference_35_39)

        csv_household_age_18_20_weights = headage_distro_csv_df['household_head_age_18_20']
        csv_household_size_4_reference_35_39 = headage_distro_csv_df['household_head_age_35_39'][5]

        self.assertTrue(csv_household_age_18_20_weights.equals(csv_household_age_18_20_weights),
                        msg="household_head_age_18_20 columns should be the same")

        self.assertEqual(csv_household_size_4_reference_35_39, dat_household_size_4_reference_35_39,
                         msg="Each cell in the data table should be identical")

# OK: contact_networks.py generate_synthetic_popluation() looks for this.
# data_distributions.py get_head_age_by_size_distr() uses this to
# call data_distributions.py get_household_head_age_by_size_df() which
# calls get_household_head_age_by_size_path() and reads as a csv into dataframe
