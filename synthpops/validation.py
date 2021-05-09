import itertools
import os
import sciris as sc
import numpy as np
import covasim as cv
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
import cmasher as cmr
import cmocean as cmo
import seaborn as sns
from scipy.spatial import distance

from . import config as cfg
from . import base as spb
from . import defaults as spd
from . import data_distributions as spdata
from . import ltcfs as spltcf
from . import households as sphh
from . import schools as spsch
from . import workplaces as spw
from . import contact_networks as spcnx
from . import pop as sppop


def get_canberra_distance(matrix1, matrix2, dim_norm=False):
    """
    Calculate the Canberra distance of two matrices.

    Args:
        matrix1 (ndarray) : first matrix
        matrix2 (ndarray) : second matrix

    Returns:
        float : The Canberra distance.
    """
    # first check the dimensions match
    dims_match = matrix1.shape == matrix2.shape
    if not dims_match:
        raise RuntimeError(f"The dimensions of the two matrices do not match ({matrix1.shape}, {matrix2.shape}).")

    # vectorize both matrices
    v1 = matrix1.reshape(matrix1.size)
    v2 = matrix2.reshape(matrix2.size)

    if dim_norm:
        return distance.canberra(v1, v2) / len(v1)
    else:
        return distance.canberra(v1, v2)
