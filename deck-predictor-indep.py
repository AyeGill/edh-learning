from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import argparse
import collections
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile
import itertools

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
tf.enable_eager_execution() 

from tensorflow.contrib.tensorboard.plugins import projector

from datasetutils import *


# Calculates M[A,B] = log P(cardA | cardB), given a dataset.
# I originally thought of this as a naive Bayes-type approach, but there is really nothing particularly Bayes about this
# We just read off the probabilities.

def decks_ids_to_prob(decks_ids, vocabsize):
    countsCommon = np.zeros([vocabsize, vocabsize]) #symmetric matrix - countsCommon[A,B] :  number of common decks for A and B
    counts = np.zeros([vocabsize]) #counts[A] : number of decks containing A
    #logctr = 0
    print("marco")
    for deck in decks_ids:
        for cardA in deck:
            counts[cardA] += 1
            for cardB in deck: #Gives bad results on duplicates!
                countsCommon[cardA,cardB] += 1
       # logctr+=1
    
    condLogProb = np.log(countsCommon) #condLogProb[A,B] = log P(A|B) = log P(A,B) - log P(B) = (log #(A,B) - log #total) - (log #B - log #total) 
                                       #= log #(A,B) - log #B
    for B in range(vocabsize):
        logcB = np.log(counts[B])
        for i in range(vocabsize):
            condLogProb[i,B] -= logcB
    return condLogProb

def test():
    decks = read_data_zip('decks.zip')
    dic,rev = read_vocab('vocabulary.txt')
    vsize = len(dic) 
    probs = decks_ids_to_prob([deck_to_ids(deck,dic) for deck in decks],vsize)
    print("A:", np.exp(probs[dic['Reclamation Sage'],dic['Forest']]))
    print("B:", np.exp(probs[dic['Roon of the Hidden Realm C'],dic['Mountain']]))
    return probs