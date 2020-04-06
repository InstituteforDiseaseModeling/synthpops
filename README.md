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

See also: `test_synthpop.py` for example usage of reading in demographic data and generating populations matching those demographics, `test_contacts.py` for example usage of generating random contact networks with individuals matching demographic data or reading in synthetic contact networks with 3 layers (households, schools, and workplaces), and `test_contact_network_generation.py` for generating synthetic contact networks  in households, schools, and workplaces with Seattle Metro data (and writing to file).

## Detailed installation instructions
1. Clone the repository. If you intend to make changes to the code, we recommend that you fork it first.
2. (Optional) Create and activate a virtual environment. 
3. Go to the root of the cloned repository and install synthpop by doing the following:

* `python setup.py develop`

The module will then be importable via `import synthpops`.

## Detailed usage

Examples live in the `examples` folder. These can be run as:

* `python examples/make_generic_contacts.py`
  
  This creates a dictionary of individuals, each of whom are represented by another dictionary with their contacts contained in the `contacts` key. Contacts are selected at random with degree distribution following the Erdos-Renyi graph model.

* `python examples/generate_contact_network_with_microstructure.py`
  
  This creates and saves to file households, schools, and workplaces of individuals with unique ids, and a table mapping ids to ages. Two versions of each contact layer (households, schools, or workplaces) are saved; one with the unique ids of each individual in each group (a single household, school or workplace), and one with their ages (for easy viewing of the age mixing patterns created).

* `python examples/load_contacts_and_show_some_layers.py`
  
  This loads a multilayer contact network made of three layers and shows the age and ages of contacts for the first 20 people. 

## Structure

All core model core in `synthpops` folder; standard usage is `import synthpops as sp`. The folder `data` contains demographic data needed and some pre-generated contact networks for populations of different sizes. 

### data
The `data` folder contains all of the demographic dat necessary to generate contact networks using `synthpops`. Please update synthpops.datadir to point at this directory.

### licenses
The `licenses` folder contains:
* `NOTICE`: Third-party software notices and information
* `notice.py`: Scraper to auto-generate the NOTICE file.

### synthpops

The `synthpops` folder contains:

* `__init__.py`
* `api.py` 
* `config.py`: Methods to set where `datadir` points; this should be the path to the data folder
* `contact_networks.py`: Functions to create a synthetic population with demographic data and places people into households, schools, and workplaces.
* `contacts.py`: Functions to create other types of contact networks and load multilayer networks.
* `plot_tools.py`: Functions to plot an age-mixing matrix for a layer in the contact network. 
* `synthpops.py`: Functions to call in demographic data and sampling functions.
* `version.py`: Version and date.

### tests

The `tests` folder contains tests of different functions available in `synthpops`.

## Disclaimer

The code in this repository was developed by researchers at IDM to support our research in infectious disease transmission in human contact networks and to explore the impact of contact tracing and testing in combination with our Covasim repository. This repository is publicly available under the Creative Commons Attribution-Noncommercial-ShareALike 4.0 License to provide other with a better understanding of our research. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. 
