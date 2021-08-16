# Author: Payam Ghassemi, payamgha@buffalo.edu
# Sep 8, 2018
# Copyright 2018 Payam Ghassemi

import numpy as np

from matplotlib import pyplot as plt

import scipy.io

import pickle

# Built-in python libraries
import sys
import os
from urllib.request import urlretrieve

# 3rd-party libraries I'll be using
import matplotlib

import pandas as pd
import seaborn as sns

from scipy import stats

#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['lines.markeredgewidth'] = 1

def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])    
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    font = {'family':'Times New Roman',
            'weight' : 'normal',
            'size'   : 12}
    matplotlib.rc("font", **font)

def set_size(fig, width=6, height=3):
    fig.set_size_inches(width, height)
    plt.tight_layout()

def get_colors():
    return np.array([
        [0.1, 0.1, 0.1],          # black
        [0.4, 0.4, 0.4],          # very dark gray
        [0.7, 0.7, 0.7],          # dark gray
        [0.9, 0.9, 0.9],          # light gray
        [0.984375, 0.7265625, 0], # dark yellow
        [1, 1, 0.9]               # light yellow
    ])


set_style()


flatui = [ "#1C366A", "#106f96", "#1DABE6", "#2ecc71", "#C3CED0", "#E43034", "#3498db", "#e74c3c","#a65d42","#6e5200","#dcc4d2"]
palette = sns.set_palette(flatui) # sns.color_palette("colorblind", 3) #"Set2"
flatui = [ "#2ecc71", "#C3CED0", "#1DABE6", "#1C366A",  "#106f96", "#E43034", "#3498db", "#e74c3c","#a65d42","#6e5200","#dcc4d2"]
palette_B = sns.set_palette(flatui) # sns.color_palette("colorblind", 3) #"Set2"