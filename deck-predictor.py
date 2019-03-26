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
input_size = 20 #Can be tuned


def get_random_cards(deck):
    #For now, output a tuple containing 20 random cards and a random card.
    cardnum = input_size+1
    cards = np.random.choice(deck, cardnum, replace=False)
    #CURRENT ERR HERE: deck is not 1-dimensional. Figure out how to map over datasets.
    return cards

def make_training_data(data):
    samples = np.apply_along_axis(get_random_cards, 1, data) #1 = dimension/Axis
    samples = np.expand_dims(samples, 1)
    data = samples[:,:,:input_size]
    labels = samples[:,:,input_size]
    print("Shapes data:", data.shape, "labels", labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    
    for dat, lab in dataset.take(1):
        print(dat,lab)

    return dataset
#----------------------------------------------

#--THE MODEL--

def build_model(vocab_size, embedding_dim, units, sample_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=20),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

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

    training_data = make_training_data(data)

    print(training_data.output_shapes)

    embedding_dim = 64 #Tune these
    units = 64*32
    model = build_model(vocab_size,embedding_dim,units,input_size)
    model.summary()

    
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS=2
    steps_per_epoch = 100
    history = model.fit(training_data.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


    def predict_card(model, partial_deck): #Stub
        cardsin = [dictionary[card] for card in partial_deck]
        assert len(cardsin)==20
        cardsin = np.expand_dims(cardsin, 0)
        predictions = np.expand_dims(model(cardsin),1)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        print(rev_dictionary[predicted_id])

    listA = ['Roon of the Hidden Realm C',
    'Birthing Pod',
    'Reclamation Sage',
    'Sol Ring',
    'Breeding Pool',
    'Deadeye Navigator',
    'Temple Garden',
    'Ixidron',
    'Ghostly Flicker',
    'Brutalizer Exarch',
    'Peregrine Drake',
    'Karmic Guide',
    'Coiling Oracle',
    'Sphinx of Uthuun',
    'Fact or Fiction',
    'Swords to Plowshares',
    'Tempt with Discovery',
    'Farhaven Elf',
    'Worldly Tutor',
    'Forest']
    print(predict_card(model,listA))




if __name__ == "__main__":
    main()
#Should store data as ids.