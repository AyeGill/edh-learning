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
#Currently hacked to just ignore duplicate cards.
#Need better solution a la Frank Karsten deck aggregation ideas

# Calculates M[A,B] = log P(cardA | cardB), given a dataset.
# I originally thought of this as a naive Bayes-type approach, but there is really nothing particularly Bayes about this
# We just read off the probabilities.

def decks_ids_to_prob(decks_ids, vocabsize):
    countsCommon = np.zeros([vocabsize, vocabsize]) #symmetric matrix - countsCommon[A,B] :  number of common decks for A and B
    counts = np.zeros([vocabsize]) #counts[A] : number of decks containing A
    #logctr = 0
    print("marco")
    for deck in decks_ids:
        for cardA in set(deck):
            counts[cardA] += 1
            for cardB in set(deck): #Gives bad results on duplicates!
                countsCommon[cardA,cardB] += 1
       # logctr+=1
    logProb = np.log(counts) - np.log(len(decks_ids))
    condLogProb = np.log(countsCommon) #condLogProb[A,B] = log P(A|B) = log P(A,B) - log P(B) = (log #(A,B) - log #total) - (log #B - log #total) 
                                       #= log #(A,B) - log #B
    for B in range(vocabsize):
        logcB = np.log(counts[B])
        for i in range(vocabsize):
            condLogProb[i,B] -= logcB
    return condLogProb,logProb

def decks_to_probs(decks_ids,vocabsize):
    counts = np.zeros([vocabsize])
    decknum = 0
    for deck in decks_ids:
        for cardA in deck:
            counts[cardA] +=1
        decknum += 1
    return np.log(counts) - np.log(decknum)

def decks_to_probs_card(cardA, decks_ids, vocabsize): #computes one column of the output of decks_ids_to_prob
    counts = np.zeros([vocabsize]) #ocurrences of each card
    countsCommon = np.zeros([vocabsize]) #occurrences of each card in a deck containing cardA
    countCardA = 0
    numdecks = 0
    for deck in decks_ids:
        common = (cardA in deck)
        if common:
            countCardA += 1
        numdecks += 1
        for card in deck:
            counts[card] +=1
            if common:
                countsCommon[card] += 1
    logProb = np.log(countsCommon) - np.log(counts)
    prob = np.log(countCardA) - np.log(numdecks)
    return logProb #logProb[card] = P(cardA | card) prob = p(cardA)
        
def predict_card_lazy(partial_list, decks_ids,vocabsize):
    outprobs = decks_to_probs(decks_ids,vocabsize)
    for card in partial_list:
        logProb = decks_to_probs_card(card,decks_ids,vocabsize)
        outprobs += logProb
        outprobs[card] = -np.inf
    return np.argmax(outprobs)


def predict_card(partial_list, probs,cprobs):
    outprobs = np.copy(probs)
    for card_id in partial_list:
        outprobs += cprobs[card_id,:]
        outprobs[card_id] = -np.inf
    return np.argmax(outprobs)

def predict_deck_lazy(partial_list, decks_ids,vocabsize):
    outprobs = decks_to_probs(decks_ids,vocabsize)
    outlist = []
    for card in partial_list:
        outprobs += decks_to_probs_card(card,decks_ids,vocabsize)
        outprobs[card] = -np.inf
    while len(outlist + partial_list)<100:
        new_card = np.argmax(outprobs)
        outlist.append(new_card)
        outprobs += decks_to_probs_card(card,decks_ids,vocabsize)
        outprobs[new_card] = -np.inf
    return outlist

def predict_deck(partial_list,probs,cprobs): #currently outputs probably too few lands.
    outlist = []
    while len(outlist + partial_list)<100:
        new_card = predict_card(partial_list + outlist, probs,cprobs)
        outlist.append(new_card)
    return outlist

def predict_deck_f(partial_list,probs,cprobs):
    outlist = []
    outprobs = np.copy(probs)
    for card_id in partial_list:
        outprobs += cprobs[card_id,:]
        outprobs[card_id] = -np.inf
    while len(outlist + partial_list)<100:
        new_card = np.argmax(outprobs)
        outlist.append(new_card)
        outprobs += cprobs[new_card,:]
        outprobs[new_card] = -np.inf
    return outlist

def load():
    dic,rev = read_vocab('vocabulary.txt')
    cprobs = np.load('cprobs.npy')
    probs = np.load('probs.npy')
    return dic,rev,cprobs,probs

def load_mmap():
    dic,rev = read_vocab('vocabulary.txt')
    cprobs = np.load('cprobs.npy',mmap_mode='r')
    probs = np.load('probs.npy',mmap_mode='r')
    return dic,rev,cprobs,probs
    
def gen_probs():
    decks = read_data_zip('decks.zip')
    dic,rev = read_vocab('vocabulary.txt')
    condprobs,probs = decks_ids_to_prob([deck_to_ids(deck,dic) for deck in decks],len(dic.keys()))
    np.save('probs',probs)
    np.save('cprobs',condprobs)