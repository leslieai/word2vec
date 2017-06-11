from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd


# reverse_dictionary = pd.read_csv('./save_dictionary.csv', header=0).values

# Step 6: Visualize the embeddings.

def plot_word(sess,embeddings):
    final_embeddings = embeddings.eval(session=sess)
    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        # plt.savefig(filename)
        plt.show()

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')


def calculate_sim(sess,reverse_dictionary, embeddings, word_id):
    valid_data = tf.constant(word_id, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(
        embeddings, valid_data)  # (1,128)
    similarity = tf.matmul(
        valid_embeddings, embeddings, transpose_b=True)  # (1,50000)
    # print(sess.run(similarity).shape)
    top_k = 8  # number of nearest neighbors
    # sim = similarity.eval(session=sess)
    # nearest = (-sim[0, :]).argsort()[1:top_k + 1]
    _,nearest =tf.nn.top_k(similarity, top_k)
    nearest = nearest.eval()
    valid_word = reverse_dictionary[word_id[0]]
    log_str = 'Nearest to %s:' % valid_word
    for k in xrange(top_k):
        close_word = reverse_dictionary[nearest[0][k]]
        log_str = '%s %s,' % (log_str, close_word)
    print(log_str)

def save_embed_visual(sess,embed_matrix):
    from tensorflow.contrib.tensorboard.plugins import projector
    import os
    # Create randomly initialized embedding weights which will be trained.
    N = 10000  # Number of items (vocab size).
    D = 200  # Dimensionality of the embedding.
    # embeddings = tf.constant(embed_matrix)
    # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))  # (1,50000)
    # normalized_embeddings = embeddings / norm  # (50000,128)
    # print(normalized_embeddings)

    # normalized_embeddings = tf.nn.l2_normalize(tf.constant(embed_matrix), 1)
    print(embed_matrix.eval())
    embedding_var = tf.Variable(embed_matrix[:10000,:], name='word_embedding')
    # embedding_var=embed_matrix[:10000, :]
    sess.run(tf.global_variables_initializer())
    print(embedding_var.eval())
    # exit()
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join('embed_visual', 'metadata.tsv')

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter('embed_visual')

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, 'embed_visual/embedding_var.ckpt', 1)
    print('save_embed_visual')
def evaluate_analogy_questions(reverse_dictionary,nemb,aid,bid,cid):
    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.constant(aid,dtype=tf.int32)  # [N]
    analogy_b = tf.constant(bid,dtype=tf.int32)  # [N]
    analogy_c = tf.constant(cid,dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    # nemb = tf.nn.l2_normalize(self._emb, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    # embedding_lookup
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs
    # print(a_emb)
    # print(b_emb)
    # print(c_emb)
    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)
    target2=tf.expand_dims(target,1)
    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    # print(nemb)
    dist = tf.matmul(nemb,target2)
    dist2=tf.transpose(dist)
    top_k=8 # number of nearest neighbors
    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist2, top_k)
    pred_idx =pred_idx.eval()
    log_str=''
    for k in xrange(top_k):
        close_word = reverse_dictionary[pred_idx[0][k]]
        log_str = ' %s,' % ( close_word)
        print(log_str)
    # print(pred_idx.eval())


def get_key_by_value(str,dict1):
    if str in dict1.values():
        print(list(dict1.keys())[list(dict1.values()).index(str)])
    else:
        print("word not found!!")
def eval():
    checkpoint_file = tf.train.latest_checkpoint('./train')
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # gg = graph.get_operations()
            # print(gg)
            embed = graph.get_operation_by_name("embed_scope/embeddings").outputs[0]

            # norm = tf.sqrt(tf.reduce_sum(tf.square(embed), 1, keep_dims=True))  # (1,50000)
            # normalized_embed = embed / norm  # (50000,128)

            normalized_embed=tf.nn.l2_normalize(embed, 1)
            # print(normalized_embed.eval())
            # exit(0)

            from data_ulit import reverse_dictionary

            # calculate_sim(sess,reverse_dictionary, normalized_embed, [24])
            # plot_word(sess,normalized_embed)
            # save_embed_visual(sess,normalized_embed)
            evaluate_analogy_questions(reverse_dictionary,normalized_embed,843,938,303)

            # a = italy, b = rome, c = france

            # get_key_by_value('italy',reverse_dictionary) #==>843
            # get_key_by_value('rome',reverse_dictionary) #==>938
            # get_key_by_value('france',reverse_dictionary) #==>303
if __name__ == '__main__':
    eval()
