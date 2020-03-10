# synthpops

Synthetic populations generation functions.

See `test_synthpop.py` for example usage.

NOTE: this module needs to load in data in order to function. To set the data location, do

```python
import synthpops
synthpops.set_datadir('my-data-folder')
```

The data folder will need to have files in this kind of structure:

```
./SyntheticPopulations
./SyntheticPopulations/setting_codes.csv
./SyntheticPopulations/synthetic_ages
./SyntheticPopulations/synthetic_ages/data_a18
./SyntheticPopulations/synthetic_ages/data_a18/a18_Shanghai.dat
./SyntheticPopulations/synthetic_ages/data_a18/a18_Hubei.dat
./SyntheticPopulations/synthetic_ages/data_a18/a18_Washington.dat
./SyntheticPopulations/setting_weights.csv
./SyntheticPopulations/asymmetric_matrices
./SyntheticPopulations/asymmetric_matrices/data_H18
./SyntheticPopulations/asymmetric_matrices/data_H18/M18_Washington_H.dat
./SyntheticPopulations/asymmetric_matrices/data_H18/M18_Hubei_H.dat
./SyntheticPopulations/asymmetric_matrices/data_H18/M18_Shanghai_H.dat
./SyntheticPopulations/asymmetric_matrices/data_S18
./SyntheticPopulations/asymmetric_matrices/data_S18/M18_Washington_S.dat
./SyntheticPopulations/asymmetric_matrices/data_S18/M18_Shanghai_S.dat
./SyntheticPopulations/asymmetric_matrices/data_S18/M18_Hubei_S.dat
./SyntheticPopulations/asymmetric_matrices/data_M18
./SyntheticPopulations/asymmetric_matrices/data_M18/M18_Shanghai_M.dat
./SyntheticPopulations/asymmetric_matrices/data_M18/M18_Washington_M.dat
./SyntheticPopulations/asymmetric_matrices/data_M18/M18_Hubei_M.dat
./SyntheticPopulations/asymmetric_matrices/data_R18
./SyntheticPopulations/asymmetric_matrices/data_R18/M18_Washington_R.dat
./SyntheticPopulations/asymmetric_matrices/data_R18/M18_Hubei_R.dat
./SyntheticPopulations/asymmetric_matrices/data_R18/M18_Shanghai_R.dat
./SyntheticPopulations/asymmetric_matrices/data_W18
./SyntheticPopulations/asymmetric_matrices/data_W18/M18_Shanghai_W.dat
./SyntheticPopulations/asymmetric_matrices/data_W18/M18_Washington_W.dat
./SyntheticPopulations/asymmetric_matrices/data_W18/M18_Hubei_W.dat
./SyntheticPopulations/README.md
./census
./census/household size distributions
./census/household size distributions/puma_householdsize_dists_byfamily_nonfamily.csv
./census/household size distributions/tract_householdsize_dists_byfamily_nonfamily.csv
./census/household size distributions/puma_householdsize_dists.csv
./census/household size distributions/tract_householdsize_dists.csv
./census/age distributions
./census/age distributions/puma_age_dists.csv
./census/age distributions/puma_age_dists_bygender.csv
./census/age distributions/seattle_metro_age_bracket_distr.dat
./census/age distributions/tract_age_dists_bygender.csv
./census/age distributions/census_age_brackets.dat
./census/age distributions/tract_age_dists.csv
./census/age distributions/seattle_metro_gender_fraction_by_age_bracket.dat
./census/household_living_arrangements
./census/household_living_arrangements/taba3.xls
./census/household_living_arrangements/README.md
./census/household_living_arrangements/tabf1-all.xls
./census/README.md
```

If that doesn't look like a thing you have access to, you probably won't be able to use this code.