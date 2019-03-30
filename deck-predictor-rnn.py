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
sequence_size = 15 #+1


def shuffle(deck):
    #For now, output a tuple containing 20 random cards and a random card.
    cards = np.random.choice(deck, 100, replace=False)
    return cards

def make_training_data(data):
    samples = np.apply_along_axis(shuffle, 1, data)
    decks, _ = samples.shape
    data = np.zeros((decks, 100-sequence_size, sequence_size))
    labels = np.zeros((decks, 100-sequence_size, sequence_size)) 
    for i in range(100-sequence_size):
        data[:,i,:] = samples[:,i:i+sequence_size]
        labels[:,i,:] = samples[:,i+1:i+sequence_size+1]
    return tf.data.Dataset.from_tensor_slices((data,labels))
#----------------------------------------------

#--THE MODEL--

def build_model(vocab_size, embedding_dim, units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,batch_input_shape=[batch_size,None]),
        tf.keras.layers.GRU(units,recurrent_activation='sigmoid',return_sequences=True),
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

    units = 16
    embedding_dim = 32 #Tune these
    batch_size = 100-sequence_size
    model = build_model(vocab_size,embedding_dim,units,batch_size)


    model.summary()

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # Loading is useless unless we save/load vocabulary /facepalm
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    def loss(labels,logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels,logits)

    model.compile(tf.train.AdamOptimizer(learning_rate=0.1) ,loss=loss) #Higher learning rate

    for input_example_batch, target_example_batch in training_data.take(1): 
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


    EPOCHS = 1
    steps_per_epoch = 50000
    model.fit(training_data.repeat(), batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,callbacks=[checkpoint_callback])

    def predict_cards(model, partial_deck):
        num_gen = 100-len(partial_deck)
        input_ids = [dictionary[card] for card in partial_deck]
        input_ids = tf.expand_dims(input_ids, 0)
        output_gen = []
        temperature = 0.1

        model.reset_states()
        for i in range(num_gen):
            predictions = model(input_ids)
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_ids = tf.expand_dims([predicted_id], 0)
            output_gen.append(rev_dictionary[predicted_id])
        return output_gen



    deckA = ['Roon of the Hidden Realm C', 'Acidic Slime', 'Angel of Finality', 'Reflector Mage', 'Eternal Witness', 'Farhaven Elf', 'Deadeye Navigator']
    print(deckA, "->", predict_cards(model,deckA))

if __name__ == "__main__":
    main()
#Should store data as ids.