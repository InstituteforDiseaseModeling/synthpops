# SynthPops

SynthPops is a module designed to generate synthetic populations that are used for COVID-19 (SARS-CoV-2) epidemic analyses. SynthPops can create generic populations with different network characteristics, as well as synthetic populations that interact in different layers of a multilayer contact network. **Note**: SynthPops is currently under active development and not all features are fully tested and documented. Currently, synthetic populations are only implemented for one region (Seattle, USA). We are in the process of expanding to include data on additional regions.


The code was originally developed to explore the impact of contact tracing and testing in human contact networks in combination with our [Covasim repository](https://github.com/InstituteforDiseaseModeling/covasim). This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.

More extensive installation and usage instructions are in the [SynthPops documentation](https://docs.idmod.org/projects/synthpops/en/latest).

## Installation

Python >=3.7, <3.9 is required. Python 2 is not supported. Virtual environments are strongly recommended but not required.

To install, first clone the GitHub repository:

`git clone https://github.com/InstituteforDiseaseModeling/synthpops.git`

Then install via:

`python setup.py develop`

Note: while `synthpops` can also be installed via [pypi](https://pypi.org/project/synthpops), this method does not currently include the data files which are required to function, and thus is not recommended. We recommend using Python virtual environments managed with [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to help with installation. Currently, our recommended installation steps are:

1. Install Anaconda. 

2. Working either in an existing conda environment or creating a new environment with Anaconda, install synthpops by navigating to the directory for this package and running `python setup.py develop` via terminal.


## Quick Start

The following code creates and plots a the household layer of a synthetic population (using defaults for Seattle, Washington):

```python
import synthpops as sp
import matplotlib.pyplot as plt

n = 10000 # how many people in your population
pop = sp.Pop(n) # create the population
pop.plot_contacts() # plot the contact matrix
plt.show() # display contact matrix to screen
```

## Usage

In addition to the [documentation](https://docs.idmod.org/projects/synthpops/en/latest/usage.html), see the `examples` folder for usage examples.

## Structure

All core modeling is in the `synthpops` folder; standard usage is `import synthpops as sp`.

### data

The `data` folder contains demographic data used by the algorithms.

### synthpops

The `synthpops` folder contains the library, including:

* `base.py`: Frequently-used functions that do not neatly fit into other areas of the code base.
* `config.py`: Methods to set general configuration options.
* `contact_networks.py`: Functions to create a synthetic population with demographic data and places people into households, schools, and workplaces.
* `data_distributions.py`: Functions for processing the data.
* `households.py`: Functions for creating household contact networks.
* `ltcfs.py`: Functions for creating long-term care facility contact networks.
* `plotting.py`: Functions to plot age-mixing matrices.
* `pop.py`: The `Pop` class, which is the foundation of SynthPops.
* `process_census.py`: Functions to process US Census data.
* `sampling.py`: Statistical sampling functions.
* `schools.py`: Functions for creating school contact networks.
* `workplaces.py`: Functions for creating workplace contact networks.

### tests

The `tests` folder contains tests of different functions available in SynthPops.


## Disclaimer

The code in this repository was developed by IDM to support our research in disease transmission and managing epidemics. We’ve made it publicly available under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License.

