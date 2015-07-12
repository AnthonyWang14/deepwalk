from collections import Counter, Mapping
from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import cpu_count
from six import string_types
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
FAST_VERSION = -1

logger = logging.getLogger("deepwalk_mssg")

from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab

def train_sentence_mssg(model, sentence, alpha, work=None):
    """
    My multi-sense-skip-gram program for one sentence
    """
    print("train sentence mssg");

    if model.negative:
        # precompute negative labels
        labels = zeros(model.negative + 1)
        labels[0] = 1.0

    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]
                neu1e = zeros(l1.shape)

                if model.hs:
                    # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                    ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                    model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error

                model.syn0[word2.index] += neu1e  # learn input -> hidden

    return len([word for word in sentence if word is not None])

class MultiSenseSkipGram(Word2Vec):

        def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
        sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0):
            print("init mssg!!!")
            self.vocab = {}  # mapping from a word (string) to a Vocab object
            self.index2word = []  # map from a word's matrix index (int) to word (string)
            self.sg = int(sg)
            self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
            self.layer1_size = int(size)
            if size % 4 != 0:
                logger.warning("consider setting layer size to a multiple of 4 for greater performance")
            self.alpha = float(alpha)
            self.window = int(window)
            self.seed = seed
            self.min_count = min_count
            self.sample = sample
            self.workers = workers
            self.min_alpha = min_alpha
            self.hs = hs
            self.negative = negative
            self.cbow_mean = int(cbow_mean)
            self.K = 3
            if sentences is not None:
                self.build_vocab(sentences)
                self.train(sentences)

        def reset_weights(self):
            print("reset weights mssg")
            """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
            logger.info("resetting layer weights")
            random.seed(self.seed)
            self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
            # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
            for i in xrange(len(self.vocab)):
                self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
            if self.hs:
                self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
            if self.negative:
                self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
            self.syn0norm = None

        def train(self, sentences, total_words=None, word_count=0, chunksize=100):
            print("train mssg")
            """
            Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
            Each sentence must be a list of unicode strings.

            """
            if not self.vocab:
                raise RuntimeError("you must first build vocabulary before training the model")

            start, next_report = time.time(), [1.0]
            word_count = [word_count]
            total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
            jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
            lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

            def worker_train():
                """Train the model, lifting lists of sentences from the jobs queue."""
                work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
                neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

                while True:
                    job = jobs.get()
                    if job is None:  # data finished, exit
                        break
                    # update the learning rate before every job
                    alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                    # how many words did we train on? out-of-vocabulary (unknown) words do not count
                    if self.sg:
                        job_words = sum(train_sentence_mssg(self, sentence, alpha, work) for sentence in job)
                    else:
                        job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                    with lock:
                        word_count[0] += job_words
                        elapsed = time.time() - start
                        if elapsed >= next_report[0]:
                            logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                                (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                            next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

            workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
            for thread in workers:
                thread.daemon = True  # make interrupting the process with ctrl+c easier
                thread.start()


