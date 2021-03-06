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

max_decks = 10000
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
            if len(decks)>=max_decks:
                break
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
    cards = np.random.choice(deck, 100, replace=False)
    #CURRENT ERR HERE: deck is not 1-dimensional. Figure out how to map over datasets.
    return cards

def make_label(cards, vocab_size):
    out = np.zeros(vocab_size)
    for c in cards:
        out[int(c)] = 1
    return out
def make_training_data(data,vocab_size):
    samples = np.apply_along_axis(get_random_cards, 1, data) #1 = dimension/Axis
    samples = np.expand_dims(samples, 1)
    data = samples[:,:,:input_size]
    labels = samples[:,:,input_size:]
    labels = np.apply_along_axis(make_label, 2, labels, vocab_size)
    print("Shapes data:", data.shape, "labels", labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((data,labels))
    
    for dat, lab in dataset.take(1):
        print(dat,lab)

    return dataset
#----------------------------------------------

#--THE MODEL--

def build_model(vocab_size, embedding_dim, units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=20),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units,activation='sigmoid'),
        tf.keras.layers.Dense(vocab_size),
        tf.keras.layers.Reshape(-1)
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
    epochIn = int(input("Train for how many epochs? :"))
    epochsTrained = int(input("Start at what epoch? :"))
    print("Training for epochs:", epochIn)
    decks = read_data_zip('decks.zip')
    numDecks = len(decks)
    print("decks:", numDecks)
    data,cards,dictionary,rev_dictionary = build_dataset(decks)
    
    vocab_size = len(cards)
    assert vocab_size==len(dictionary)
    print("Data:")
    print(data.shape)
    print(data[10,10])
    print(data[1010, 5])
    if epochIn>0:
        training_data = make_training_data(data,vocab_size)

<<<<<<< HEAD
    embedding_dim = 16 #Tune these
    units = 32
    model = build_model(vocab_size,embedding_dim,units,input_size)
    model.summary()

    
    def loss(labels, logits):
        labels = tf.expand_dims(labels, 0)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    #if epochsTrained==0:
    #    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    if epochIn>0:
        EPOCHS=epochIn
        steps_per_epoch = 10000
        history = model.fit(training_data.repeat(), epochs=EPOCHS, initial_epoch=epochsTrained, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])


    def predict_card(model, partial_deck): #Stub
        cardsin = [dictionary[card] for card in partial_deck]
        assert len(cardsin)==20
        cardsin = np.expand_dims(cardsin, 0)
        predictions = np.expand_dims(model(cardsin),1)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        return rev_dictionary[predicted_id]

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

    listB = ['Roon of the Hidden Realm C',
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
    'Plains']
    for i in range(10):
        print(predict_card(model,listA), predict_card(model,listB))
=======
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

>>>>>>> 93f39be732f85619bc10927c8f3abfc165f9619a

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