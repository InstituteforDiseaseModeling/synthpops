# SynthPops

SynthPops is a module designed to generate synthetic populations that are used for COVID-19 (SARS-CoV-2) epidemic analyses. SynthPops can create generic populations with different network characteristics, as well as synthetic populations that interact in different layers of a multilayer contact network.

More extensive installation and usage instructions are in the [SynthPops documentation](https://institutefordiseasemodeling.github.io/synthpops/).

## Installation

Python >=3.6 is required. Python 2 is not supported. Virtual environments are recommended but not required.

`python setup.py develop`

NOTE: This module needs to load in data in order to function. To set the data location, do

```python
import synthpops as sp
sp.set_datadir('my-data-folder')
```
The data folder will need to have files in this kind of structure:

```bash
demographics/
contact_matrices_152_countries/
```
You can find this data under the data folder.

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

The code in this repository was developed by researchers at IDM to support our research in infectious disease transmission in human contact networks and to explore the impact of contact tracing and testing in combination with our [Covasim repository](https://github.com/InstituteforDiseaseModeling/covasim). This repository is publicly available under the Creative Commons Attribution-Noncommercial-ShareALike 4.0 License to provide other with a better understanding of our research. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests.
