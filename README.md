# SynthPops

SynthPops is a module designed to generate synthetic populations that are used for COVID-19 (SARS-CoV-2) epidemic analyses. SynthPops can create generic populations with different network characteristics, as well as synthetic populations that interact in different layers of a multilayer contact network. **Note**: SynthPops is currently under active development and most features are fully tested and documented, but not all. We are in the process of expanding to include data and validation on additional regions beyond the original scope of the Seattle-King County region of Washington, USA. At the moment we have data for the following locations (in the synthpops/data folder) :

* Seattle Metro, Washington, USA
* Spokane County, Washington, USA
* Franklin County, Washington, USA
* Island County, Washington, USA
* Dakar, Dakar Region, Senegal
* Zimbabwe\*
* Malawi\*
* Nepal\*

\* Data for these locations are at the national scale. In the future, we hope to provide data at a more fine grained resolution for these locations.


The code was originally developed to explore the impact of contact tracing and testing in human contact networks in combination with our [Covasim repository](https://github.com/InstituteforDiseaseModeling/covasim). This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.

More extensive installation and usage instructions are in the [SynthPops documentation](https://docs.idmod.org/projects/synthpops/en/latest).

A scientific manuscript describing the model is currently in progress. If you use the model, in the mean time the recommended citation is:

**SynthPops: a generative model of human contact networks**. Mistry D, Kerr CC, Abeysuriya R, Wu M, Fisher M, Thompson A, Skrip L, Cohen JA, Althouse BM, Klein DJ (2021). (in preparation). 


## Installation

Python >=3.7, <3.9 is required. Python 2 is not supported. Virtual environments are strongly recommended but not required.

To install, first clone the GitHub repository:

`git clone https://github.com/InstituteforDiseaseModeling/synthpops.git`

Then install via:

`python setup.py develop`

Note: while `synthpops` can also be installed via [pypi](https://pypi.org/project/synthpops), this method does not currently include the data files which are required to function, and thus is not recommended. We recommend using Python virtual environments managed with [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to help with installation. Currently, our recommended installation steps are:

1. Install Anaconda.
2. Working either in an existing conda environment or creating a new environment with Anaconda, verify that you are running an acceptable version of python (currently >=3.7, <3.9). To create a new environment: `conda create -n synthpops python=3.8.11 anaconda`. The argument to `-n` is the name for your environment. You can get a list of the available python versions (for `python=X.Y.Z` with `conda search python`.
3. If you created a new environment, activate it with `conda activate synthpops` (or whatever you named your environment).
4. Install synthpops by navigating to the directory for this package and running `python setup.py develop` via terminal. The installation process may fail with missing packages, add those using `pip` or you could start with this set, and then run the setup command: `pip install jsonobject cmasher cmocean graphviz pydot sciris`.

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

The code in this repository was developed by IDM to support our research in disease transmission and managing epidemics. We’ve made it publicly available under the Creative Commons Attribution-ShareAlike 4.0 International License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the Creative Commons Attribution-Noncommercial-ShareAlike 4.0 License.

