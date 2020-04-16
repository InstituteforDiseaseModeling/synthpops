=========
synthpops
=========

Synthetic populations generation functions.

Installation
============

`python setup.py develop`

Requirements
============

NOTE: this module needs to load in data in order to function. To set the data location, do

```python
import synthpops
synthpops.set_datadir('my-data-folder')
```

The data folder will need to have files in this kind of structure:

```bash
census/
demographics/
mortality_age_brackets.dat
mortality_rates_by_age_bracket.dat
SyntheticPopulations/
```

If that doesn't look like a thing you have access to, you probably won't be able to use this code.

Usage
=====

See `test_synthpop.py` for example usage.
