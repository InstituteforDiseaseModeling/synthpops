# SynthPops

SynthPops is a module designed to generate synthetic populations that are used for COVID-19 (SARS-CoV-2) epidemic analyses. SynthPops can create generic populations with different network characteristics, as well as synthetic populations that interact in different layers of a multilayer contact network. **Note**: SynthPops is currently under active development and not all features are fully tested and documented. Currently, synthetic populations are only implemented for one region (Seattle, USA). We are in the process of expanding to include data on additional regions.


The code was developed to explore the impact of contact tracing and testing in human contact networks in combination with our [Covasim repository](https://github.com/InstituteforDiseaseModeling/covasim).

More extensive installation and usage instructions are in the [SynthPops documentation](https://institutefordiseasemodeling.github.io/synthpops/).

## Installation

Python >=3.6 is required. Python 2 is not supported. Virtual environments are recommended but not required.

To install, first clone the GitHub repository, and then type:

`python setup.py develop`

Note: while `synthpops` can also be installed via pypi, this method does not currently include the data files which are required to function, and thus is not recommended.

## Quick Start

The following code creates a synthetic population for Seattle, Washington::
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
sp.generate_synthetic_population(npop,datadir,location=location,
                                 state_location=state_location,country_location=country_location,
                                 sheet_name=sheet_name,level=level)
```

## Usage

In addition to the [documentation](https://institutefordiseasemodeling.github.io/synthpops/usage.html), see the `examples` folder for usage examples.

## Structure

All core modeling is in the `synthpops` folder; standard usage is `import synthpops as sp`.

### data

The `data` folder contains demographic data needed and some pre-generated contact networks for populations of different sizes. Please update synthpops.datadir to point at this directory.

### licenses

The `licenses` folder contains:
* `NOTICE`: Third-party software notices and information
* `notice.py`: Scraper to auto-generate the NOTICE file.

### synthpops

The `synthpops` folder contains:

* `__init__.py`
* `api.py`
* `config.py`: Methods to set where `datadir` points; this should be the path to the data folder.
* `contact_networks.py`: Functions to create a synthetic population with demographic data and places people into households, schools, and workplaces.
* `contacts.py`: Functions to create other types of contact networks and load multilayer networks.
* `plot_tools.py`: Functions to plot an age-mixing matrix for a layer in the contact network.
* `synthpops.py`: Functions to call in demographic data and sampling functions.
* `version.py`: Version and date.

### tests

The `tests` folder contains tests of different functions available in SynthPops.

## Disclaimer


The code in this repository was developed by IDM to support our research in disease transmission and managing epidemics. We’ve made it publicly available under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License.

