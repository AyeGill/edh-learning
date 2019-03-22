from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

def read_data(dir):
        """Read lines from all files in dir"""
        data = []
        for filename in os.listdir(dir):
            with open(dir + filename) as f:
                data += [[line.strip() for line in f.readlines()]]
        return data


## Should read vocabulary from word embedding.