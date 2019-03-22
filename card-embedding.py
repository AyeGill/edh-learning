"""Basic word2vec example, modified to load commander cards"""

## COMMENT 2019-03-20: The current, 64-dimensional implementation,
## with simple two-card training sets, seems to work fairly well.
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

data_index = 0


def word2vec_basic(log_dir):
    """Example of building, training and visualizing a word2vec model."""
    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Step 1: Download the data.
    url = 'http://mattmahoney.net/dc/'

    # pylint: disable=redefined-outer-name

    # Read the data into a list of lists of strings
    def read_data_dir(dir):
        """Read lines from all files in dir"""
        data = []
        for filename in os.listdir(dir):
            with open(dir + filename) as f:
                data += [[line.strip() for line in f.readlines()]]
        return data
    def read_data_zip(filename):
        """Read lines from all files in filename (which should be zip)"""
        data = []
        print(filename)
        with zipfile.ZipFile(filename) as z:
            for deckname in z.infolist():
                with z.open(deckname) as f:
                    data.append([line.strip().decode("utf-8") for line in f.readlines()])
        return data
    
    dataDir = "decks/"
    dataZip = 'decks.zip'
    vocabulary = read_data_zip(dataZip)
    print('Decks', len(vocabulary))

    print(vocabulary[10])
    # Step 2: Build the dictionary and replace rare words with UNK token.

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
        print(dictionary[b'1 Jodah, Archmage Eternal C'])
        print(dictionary[b'1 District Guide'])
        data = list()
        unk_count = 0
        i = 0
        for deck in decks:
            if len(deck)!=0:
                data.append(list())
                for card in deck:
                    index = dictionary.get(card, 0)
                    data[i].append(index)
                i+=1
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, cards, dictionary, reversed_dictionary

    # Filling 4 global variables:
    # data - list of lists of decks, each card entry replaced by code.
    # count - map of cards(strings) to count of occurrences
    # dictionary - map of cards(strings) to their codes(integers)
    # reverse_dictionary - maps codes(integers) to cards(strings)
    data, count, unused_dictionary, reverse_dictionary = build_dataset(
        vocabulary)
    vocabulary_size = len(reverse_dictionary.keys())
    del vocabulary  # Hint to reduce memory.
    print('Sample data', data[0][:10], [reverse_dictionary[i] for i in data[0][:10]])
    for deck in data:
        if len(deck)==0:
            print("zero deck found")
    for i in range(len(reverse_dictionary.keys())):
        if not(i in reverse_dictionary):
            print(i,"not in dict, what gives?")
    
    # Step 3: Function to generate a training batch for the skip-gram model.
    # Try to predict another card in the deck given a card.
    def generate_batch(batch_size):
        global data_index
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        if data_index + 2 >= len(data):
            data_index = 0
        
        data_index += 1
        for i in range(batch_size):
            while (len(data[data_index])<2):
                print("stuck on short deck")
                data_index += 1
                if data_index==len(data):
                    data_index = 0
            context_cards = range(len(data[data_index]))
            cards = random.sample(context_cards, 2)
            batch[i] = data[data_index][cards[0]]
            labels[i,0] = data[data_index][cards[1]]
            if data_index+1>=len(data):
                data_index = 0
            else:
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        return batch, labels

    batch, labels = generate_batch(batch_size=8)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
            reverse_dictionary[labels[i, 0]])
     # Step 4: Build and train a skip-gram model.

    batch_size = 128
    embedding_size = 64  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit
    # the validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    print(valid_examples)
    graph = tf.Graph()

    with graph.as_default():

    # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

        print("Marco!")
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

        print("Polo!")

    # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
        similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
        merged = tf.summary.merge_all()

    # Add variable initializer.
        init = tf.global_variables_initializer()

    # Create a saver.
        saver = tf.train.Saver()

  # Step 5: Begin training.
    print("Step 5: Begin training")
    num_steps = 100001

    with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # Define metadata variable.
            run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned
      # "summary" variable. Feed metadata variable to session for visualizing
      # the graph in TensorBoard.
            _, summary, loss_val = session.run([optimizer, merged, loss],
                                         feed_dict=feed_dict,
                                         run_metadata=run_metadata)
            average_loss += loss_val

      # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        print("Done stepping")
        final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
        with open(log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(writer, config)

    writer.close()
    print("Writer closed")

  # Step 6: Visualize the embeddings.

  # pylint: disable=missing-docstring
  # Function to draw visualization of distance between embeddings.
    print("Plotting:")
    def plot_with_labels(low_dim_embs, labels, filename):
        print("Calling plotwlabels")
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
        plt.savefig(filename)

    try:
    # pylint: disable=g-import-not-at-top
        print("In plot block")
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(),
                                                        'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)
    


# All functionality is run after tf.app.run() (b/122547914). This could be split
# up but the methods are laid sequentially with their usage for clarity.
def main(unused_argv):
  # Give a folder path as an argument with '--log_dir' to save
  # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()
    word2vec_basic(flags.log_dir)

if __name__ == '__main__':
    tf.app.run()