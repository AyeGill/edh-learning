# EDH-learning

A project in using tensorflow to study edh decks.

Note that the present state of the data needs some sanitizing, to correct the following problems:

- 0s of cards appear
- Certain cards have metadata in brackets, e.g. `15 [4E] Swamp` and `1 Abyssal Specter [7ED] (F)`.
- Uncards appear on certain lists.

More problems may exist.

`card-embedding.py` was modified from the basic `word2vec.py` example, available on [github](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py).

It produces a word embedding for cardnames.
Cardnames are read from the decklists, one line at a time.
There is no interpretation of lines (except that trailing whitespace is stripped),
so for instance the lines `1 Forest` and `2 Forest` are treated as completely independent cards.

Future projects: train a model to complete decks.