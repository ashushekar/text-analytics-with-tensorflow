"""
RNN for Text Sequences using TensorFlow
"""
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
gendata.createtextdata()
gendata.createlabels()

# Split the dataset into train and test set
splitteddataset = gendata.splitdataset()
trainX, trainY, testX, testY, trainSeq, testSeq = splitteddataset