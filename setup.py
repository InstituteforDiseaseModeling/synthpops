import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'synthpops', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="synthpops",
    version=version,
    author="Dina Mistry",
    author_email="covid@idmod.org",
    description="Synthetic contact network generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://synthpops.org',
    keywords=['human contact networks', 'synthetic population', 'census', 'demography'],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "sciris>=0.16.5",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "numba==0.48.0",
        "networkx",
        "cmocean",
        "cmasher",
        "seaborn",
        "graphviz",
        "pydot",
    ],
)
