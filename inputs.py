from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import spacy
from nltk.tokenize import word_tokenize
import tensorflow as tf
nlp = spacy.load("en")

BATCH_SIZE = 128
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 64
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH = 10000
FRACTION = 0.4
MIN_AFTER_DEQUEUE = int(NUM_EXAMPLES_PER_EPOCH * FRACTION)


def word2id(word):
    return nlp.vocab.strings[word]


def id2word(id):
    return nlp.vocab.strings[id]


def sent2ids(sent):
    return map(word2id, word_tokenize(sent))


def ids2sent(ids):
    return " ".join(map(id2word, ids))


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example = tf.parse_single_example(serialized=serialized_example,
                                      features={
                                          'input': None,
                                          'label': None
                                      })
    input = example['input']
    label = example['label']
    return input, label


def records_inputs(input, label):
    inputs, labels = tf.train.shuffle_batch([input, label],
                                            batch_size=BATCH_SIZE,
                                            capacity=MIN_AFTER_DEQUEUE + 3 * BATCH_SIZE,
                                            min_after_dequeue=MIN_AFTER_DEQUEUE)
    return inputs, labels


class Inputs(object):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.vocab_size = VOCAB_SIZE
        self.sequence_length = SEQUENCE_LENGTH
        self.num_classes = NUM_CLASSES