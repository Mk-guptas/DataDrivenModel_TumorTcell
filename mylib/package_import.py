# this import librarires


from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
from scipy.optimize import least_squares
import pickle
import math
import time
from itertools import combinations
from scipy.stats.qmc import Sobol



from functools import partial 


import concurrent.futures as _cf
from joblib import Parallel, delayed

from scipy.integrate import odeint as scipy_odeint
from scipy.integrate import solve_ivp

from scipy.optimize import least_squares
from scipy.optimize import basinhopping

import datetime
import os
import pickle as pkl
from scipy.stats import beta
from scipy.stats import truncnorm
from scipy.stats import gamma
from scipy.stats import truncexpon

from functools import partial 

import torch
import torch.optim as optim
from torchdiffeq import odeint as torch_odeint
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # stdlib types/utilities
    "dataclass", "Tuple", "List",
    "datetime", "math", "os", "pathlib", "time",
    "partial", "combinations",

    # scientific stack
    "np", "pd", "plt",

    # joblib / parallel
    "Parallel", "delayed",
    "_cf",  # note: leading underscore requires being listed in __all__

    # SciPy
    "scipy_odeint", "solve_ivp",
    "least_squares", "basinhopping",
    "Sobol",
    "beta", "truncnorm", "gamma", "truncexpon",

    # serialization
    "pickle", "pkl",

    # PyTorch
    "torch", "nn", "F", "optim", "torch_odeint",
]
