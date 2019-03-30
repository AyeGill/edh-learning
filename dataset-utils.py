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
import itertools

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.contrib.tensorboard.plugins import projector



def read_data_zip(filename):
    #This should really operate on codes, not strings.
    """Read lines from all files in filename (which should be zip)"""
    decks = []
    print(filename)
    with zipfile.ZipFile(filename) as z:
        for deckname in z.infolist():
            with z.open(deckname) as f:
                lines = f.readlines()
                if len(lines)==100:
                    decks.append([line.strip().decode("utf-8") for line in lines])
                else:
                    print("ERR: deck with wrong size", deckname, "not loaded, moving on")
    return decks

def gen_vocab(decks):
    cards = set()
    for deck in decks:
        cards.update(deck)
    print("cards:", len(cards))
    dictionary = dict()
    ctr = 0
    for word in list(cards):
        dictionary[word] = ctr
        ctr += 1
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary
    #dictionary: cards->ints | reversed_dictionary: ints->cards

def save_vocab(dictionary, filename):
    with open(filename,'w') as f:
        for key in dictionary:
            f.write(str(key))
            f.write('\n')

def read_vocab(filename):
    dictionary = dict()
    with open(filename) as f:
        ctr = 0
        for line in f.readlines():
            dictionary[line.strip()] = ctr
            ctr += 1
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    print("cards:", len(dictionary.keys()))
    return dictionary, reversed_dictionary

def deck_to_ids(deck,dictionary):
    return [dictionary[card] for card in deck]

def deck_from_ids(deck_ids,reversed_dictionary):
    return [reversed_dictionary[id] for id in deck_ids]

def deck_to_vector(deck_ids, vocabsize): #Think about dealing with multiples
    vec = np.zeros(vocabsize)
    for card_id in deck_ids:
        vec[card_id] = 1
    return vec

def deck_vectors_dataset(decks_ids,vocabsize): #rewrite to use tf.Dataset from beginning?
    def gen():
        for deck in decks_ids:
            yield deck_to_vector(deck,vocabsize)
    dataset = tf.data.Dataset.from_generator(#consider types?
        gen, tf.float32, tf.TensorShape([vocabsize])
    )
    return dataset

def test_setup():
    decks = read_data_zip('decks.zip')
    dic, rev = gen_vocab(decks)
    return decks, dic,rev

def test_save():
    _, dic,rev = test_setup()
    save_vocab(dic, 'vocabulary.txt')
    dic2, rev2 = read_vocab('vocabulary.txt')
    assert dic==dic2
    assert rev==rev2

def test_dataset(decks,dic):
    vocabsize = len(dic.keys())
    ds = deck_vectors_dataset([deck_to_ids(deck, dic) for deck in decks], vocabsize)
    for d in ds.take(2):
        print(str(d)[:10])

