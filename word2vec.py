# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# import os
# import random
# import zipfile
# from six.moves import urllib
import math
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd

from data_ulit import generate_batch,reverse_dictionary

# reverse_dictionary = pd.read_csv('./save_dictionary.csv', header=0).to_dict(orient='dict')['reverse_dictionary']


# print(reverse_dictionary)
# print(reverse_dictionary.shape)
# exit()

# Step 4: Build and train a skip-gram model.


class SkipGramModel:
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope("data"):
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name='train_inputs')
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='train_labels')

    def _create_embedding(self):
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            with tf.name_scope("embed_scope"):
                # Look up embeddings for inputs.
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='embeddings')
                # Step 3: define the inference
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs, name='embed')

    def _create_loss(self):
        with tf.name_scope("variables"):
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embed_size],
                                    stddev=1.0 / math.sqrt(self.embed_size)), name='nce_weights')
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]), name='nce_biases')

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.train_labels,
                               inputs=self.embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size), name='nce_loss')

    def _create_optimizer(self):
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


# Step 5: Begin training.

def train_model(model,batch_size, num_skips, skip_window):
    num_steps = 100001
    # graph = tf.Graph()
    # with graph.as_default():
    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter('./train', session.graph)

        session.run(tf.global_variables_initializer())


        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16  # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name='valid_dataset')

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(model.embeddings), 1, keep_dims=True))  # (1,50000)
        normalized_embeddings = model.embeddings / norm  # (50000,128)
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)  # (16,128)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)  # (16,50000)


        print('Initialized')

        # top_k = 8  # number of nearest neighbors
        # sim=similarity.eval(session=session)
        # nearest = (-sim[0, :]).argsort()[1:top_k + 1]
        # valid_word = reverse_dictionary[valid_examples[i]]
        # log_str = 'Nearest to %s:' % valid_word
        # for k in xrange(top_k):
        #     close_word = reverse_dictionary[nearest[k]]
        #     log_str = '%s %s,' % (log_str, close_word)
        # print(log_str)
        # exit(code=0)

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, train_step = session.run([model.train_op, model.loss, model.global_step], feed_dict=feed_dict)
            current_step = tf.train.global_step(session, model.global_step)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
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


        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = model.embeddings.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = os.path.join('./embed_visual', 'metadata.tsv')

        # Use the same LOG_DIR where you stored your checkpoint.
        # summary_writer = tf.summary.FileWriter('train')

        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([model.embeddings])
        saver_embed.save(session, './train/embed_matrix.ckpt', 1)
        # final_embeddings = normalized_embeddings.eval()
        # print(final_embeddings)


VOCABULARY_SIZE = 50000
BATCH_SIZE = 128
EMBEDDING_SIZE = 128  # Dimension of the embedding vector.
SKIP_WINDOW = 1  # How many words to consider left and right.
NUM_SKIPS = 2  # How many times to reuse an input to generate a label.
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0


def main():
    model = SkipGramModel(VOCABULARY_SIZE, EMBEDDING_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    train_model(model,BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW )


if __name__ == '__main__':
    main()
