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

def read_data_zip(filename):
    #This should really operate on codes, not strings.
    """Read lines from all files in filename (which should be zip)"""
    decks = []
    print(filename)
    with zipfile.ZipFile(filename) as z:
        for deckname in z.infolist():
            with z.open(deckname) as f:
                decks.append([line.strip().decode("utf-8") for line in f.readlines()])
    return decks

def build_dataset(decks):
    """Process raw inputs into a dataset."""
    cards = set()
    # print(decks[0])
    for deck in decks:
        cards.update(deck)
    dictionary = dict()
    ctr = 0
    for word in list(cards):
        dictionary[word] = ctr
        ctr += 1
    print("lens", ctr, len(cards), len(dictionary.keys()),len(dictionary.values()))
    print("Jodah:", dictionary['Jodah, Archmage Eternal C'])
    print("District Guide:", dictionary['District Guide'])
    data = list()
    i = 0
    for deck in decks:
        if len(deck)!=0:
            data.append(list())
            for card in deck:
                index = dictionary.get(card, 0)
                data[i].append(index)
            i+=1
        else:
            print("ERR: Found empty deck")
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, cards, dictionary, reversed_dictionary
#data is a list of lists of ids(integers).
#cards is a set of card names
#dictionary is a dict mapping names to ids.
#reversed_dictionary is a dict mapping ids to names.

def predict_card(partial_deck): #Stub
    return 10

def complete_deck(partial_deck):
    curr_deck = partial_deck
    while len(curr_deck)<100:
        curr_deck.append(predict_card(curr_deck))
    print("Deck completen, length:", len(curr_deck))
    return curr_deck

def main():
    decks = read_data_zip('decks.zip')
    data,cards,dictionary,rev_dictionary = build_dataset(decks)
    
    def complete_deck_str(part_deck_str):
        part_deck_codes = [dictionary[name] for name in part_deck_str]
        comp_deck_codes = complete_deck(part_deck_codes)
        return [rev_dictionary[code] for code in comp_deck_codes]

    deckA = ['Jodah, Archmage Eternal C']
    deckB = ['Thrasios, Triton Hero C', 'Birthing Pod']

    print(deckA, "->", complete_deck_str(deckA))
    print(deckB, "->", complete_deck_str(deckB))

if __name__ == "__main__":
    main()
#Should store data as ids.