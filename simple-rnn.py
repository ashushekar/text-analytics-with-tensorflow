"""
RNN for Text Sequences using TensorFlow
"""
import numpy as np
import tensorflow as tf
from datagenerator import Generate
print(tf.__version__)

BATCHSIZE = 128
EMBEDDINGDIMENSION = 64
NUMCLASSES = 2
HIDDENLAYERSIZE = 32
TIMESTEPS = 6
ELEMENTSIZE = 1

# Generate dataset and labels
gendata = Generate(10000)
gendata.word2index()
gendata.index2word()
gendata.createtextdata()
gendata.createlabels()

# Split the dataset into train and test set
splitteddataset = gendata.splitdataset()
trainX, trainY, testX, testY, trainSeq, testSeq = splitteddataset


def getnextbatch(batchsize, dataX, dataY, data_seqlens):
    """
    A function that generates batches of sentences. Each sentence in a batch is simply a
    list of integer IDs corresponding to words.
    """
    indices = list(range(len(dataX)))
    np.random.shuffle(indices)
    batch = indices[:batchsize]
    x = [[gendata.word2index_map[word] for word in dataX[i].lower().split()]
         for i in batch]
    y = [dataY[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens

# Create placeholders for dataset
_inputs = tf.placeholder(tf.int32, shape=[BATCHSIZE, TIMESTEPS])
_labels = tf.placeholder(tf.float32, shape=[BATCHSIZE, NUMCLASSES])

# sequencelens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[BATCHSIZE])

    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_normal([len(gendata.index2word_map),
                                                   EMBEDDINGDIMENSION], -1.0, 1.0),
                                 name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, _inputs)

