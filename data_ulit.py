from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import zipfile
from six.moves import urllib
import pandas as pd
import tensorflow as tf
import numpy as np

# Step 1: Download the data.

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve( filename, 'text8.zip')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# filename = maybe_download('http://mattmahoney.net/dc/text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # data = tf.compat.as_str('The quick brown fox jumps over the lazy dog').split()
        # data = tf.compat.as_str('''
        # The problem is that generative models don’t work well in practice. At least not yet. Because they have so much freedom in how they can respond, generative models tend to make grammatical mistakes and produce irrelevant, generic or inconsistent responses. They also need huge amounts of training data and are hard to optimize. The vast majority of production systems today are retrieval-based, or a combination of retrieval-based and generative. Google’s Smart Reply is a good example. Generative models are an active area of research, but we’re not quite there yet. If you want to build a conversational agent today your best bet is most likely a retrieval-based model.''').split()
    return data



# Step 2: Build the dictionary and replace rare words with UNK token.



def build_dataset(words, n_words):

    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


filename = '../my_data/text8.zip'
vocabulary_size = 50000
vocabulary = read_data(filename)
print('Data size', len(vocabulary))
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size)
del vocabulary,dictionary  # Hint to reduce memory.

# print(reverse_dictionary) #==>{'0':'the','1':'at'}
# print(count) #==>[['UNK', -1],('the',111212),('at',2132)]
# print(data) #==>[213,3,2,13,12,32,13,3]


# print(reverse_dictionary) #dict
# exit()
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
def write_word_frequency():
    with open('embed_visual/metadata.tsv','w') as f:
        f.write("Word\tFrequency\n")
        # for index,label in enumerate(count):
        for w in count:
            f.write("%s\t%d\n" % (w[0],w[1]))
# write_word_frequency()

# save_data = pd.DataFrame({
#         "data": data,
#     })
# save_data.to_csv('save_data.csv', index=False)
# save_dictionary = pd.DataFrame({
#         "reverse_dictionary": reverse_dictionary,
#     })
# save_dictionary.to_csv('save_dictionary.csv', index=False)
#





# Step 3: Function to generate a training batch for the skip-gram model.
data_index = 0

# data = pd.read_csv('./save_data.csv', header=0).values

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # print(data_index)
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# batch, labels = generate_batch(batch_size=8, num_skips=1, skip_window=1)
# # print(batch)
# # print(labels)
#
# for i in range(8):
#     print(batch[i], reverse_dictionary[batch[i]],
#           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# exit()

