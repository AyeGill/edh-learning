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

def build_dataset(decks):
    #Note: this will fail miserably if decks contains empty decks!
    """Process raw inputs into a dataset."""
    cards = set()
    # print(decks[0])
    for deck in decks:
        cards.update(deck)
    print("cards:", len(cards))
    dictionary = dict()
    ctr = 0
    for word in list(cards):
        dictionary[word] = ctr
        ctr += 1
    print("lens", ctr, len(cards), len(dictionary.keys()),len(dictionary.values()))
    print("Jodah:", dictionary['Jodah, Archmage Eternal C'])
    print("District Guide:", dictionary['District Guide'])
    data = np.zeros((len(decks),100))
    i = 0
    for deck in decks:
        j = 0
        for card in deck:
            index = dictionary[card]
            data[i,j] = index
            j+=1
        i+=1

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, cards, dictionary, reversed_dictionary
#data is a numpy array. Shape is ($number of decks,100). Contains the decks in the obvious way.
#cards is a set of card names
#dictionary is a dict mapping names to ids.
#reversed_dictionary is a dict mapping ids to names.


#--THE PREDICTION TASK--
#The task: given a random subset of a deck, predict another card from the deck
#For now, the last card is randomized.

deck_size = 100
input_size = 2 #Can be tuned - +1


def get_random_cards(deck):
    #For now, output a tuple containing 20 random cards and a random card.
    cardnum = input_size+1
    cards = np.random.choice(deck, cardnum, replace=False)
    #CURRENT ERR HERE: deck is not 1-dimensional. Figure out how to map over datasets.
    return cards

def make_training_data(data):
    samples = np.apply_along_axis(get_random_cards, 1, data)
    data = samples[:,:input_size]
    labels = samples[:,input_size]
    return data,labels
#----------------------------------------------

#--THE MODEL--

def build_model(vocab_size, embedding_dim, units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
        tf.keras.layers.Dense(embedding_dim),
        tf.keras.layers.Dense(units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

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
    print("decks:", len(decks))
    data,cards,dictionary,rev_dictionary = build_dataset(decks)
    
    vocab_size = len(cards)
    assert vocab_size==len(dictionary)
    print("Data:")
    print(data.shape)
    print(data[10,10])
    print(data[1010, 5])

    training_data, labels = make_training_data(data)

    print("Training data:")
    print(training_data.shape)
    print("Labels:")
    print(labels.shape)
    embedding_dim = 64 #Tune these
    units = 64
    batch_size = 64
    model = build_model(vocab_size,embedding_dim,units,batch_size)
    model.summary()

    for input_example_batch, target_example_batch in training_data, labels: 
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


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