# synthpops

Synthetic populations generation functions.

## Installation

`python setup.py develop`

## Requirements

Python 3 supported.

NOTE: this module needs to load in data in order to function. To set the data location, do

```python
import synthpops
synthpops.set_datadir('my-data-folder')
```

The data folder will need to have files in this kind of structure:

```bash
demographics/
contact_matrices_152_countries/
```
You can find this data under the data folder. 

If that doesn't look like a thing you have access to, you probably won't be able to use this code.

## Quick start guide

```python
import synthpops as sp

sp.validate()

datadir = sp.datadir # this should be where your demographics data folder resides

location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'
sheet_name = 'United States of America'
level = 'county'

npop = 10000 # how many people in your population
sp.generate_synthetic_population(npop,datadir,location=location,state_location=state_location,country_location=country_location,sheet_name=sheet_name,level=level)

```

See also: `test_synthpop.py` for example usage.
