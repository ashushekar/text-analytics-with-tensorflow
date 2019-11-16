# RNN for Text Sequences using TensorFlow

Text data has some properties distinctly different from image data. These properties can make it somewhat difficult to
handle text data at first, and text data always requires at least some basic preprocessing steps for us to be able to work 
with it. To introduce working with text in TensorFlow, we will thus focus on the core components and create a minimal, 
contrived text dataset that will let us get straight to the action.

## Text Sequences

Consider the following sentence: “Our company provides smart agriculture solutions for farms, with advanced AI, 
deep-learning.” Say we obtain this sentence from an online news blog, and wish to process it as part of our machine 
learning system.

Each of the words in this sentence would be represented with an _ID_—an integer, commonly referred to as a token _ID_ in
NLP. So, the word “agriculture” could, for instance, be mapped to the integer 3452, the word “farm” to 12, “AI” to 150, 
and “deep-learning” to 0. This representation in terms of integer identifiers is very different from the vector of pixels 
in image data, in multiple ways. To make things more concrete, let us start by creating our simplified text data.

Our simulated data consists of two classes of very short “sentences,” one composed of odd digits and the other of even 
digits (with numbers written in English). We generate sentences built of words representing even and odd numbers. Our goal 
is to learn to classify each sentence as either odd or even in a supervised text-classification task.

Of course, we do not really need any machine learning for this simple task—we use this contrived example only for 
illustrative purposes.

We create sentences. We sample random digits and map them to the corresponding “words” (e.g., 1 is mapped to “One,” 7 to 
“Seven,” etc.). Text sequences typically have variable lengths, which is of course the case for all real natural language 
data. To make our simulated sentences have different lengths, we sample for each sentence a random length between 3 and 6 
with np.random.choice(range(3, 7))—the lower bound is inclusive, and the upper bound is exclusive.

Now, to put all our input sentences in one tensor (per batch of data instances), we need them to somehow be of the same 
size—so we pad sentences with a length shorter than 6 with zeros (or PAD symbols) to make all sentences equally sized 
(artificially). This pre-processing step is known as **zero-padding**.

```python
import numpy as np

class Generate:
    """
    Class to create object of text data
    """
    def __init__(self):
        self.digit_to_word_map = {0: "PAD", 1: "One", 2: "Two", 3: "Three",
                                  4: "Four", 5: "Five", 6: "Six",
                                  7: "Seven", 8: "Eight", 9: "Nine"}
        self.even_sentences = []
        self.odd_sentences = []
        self.seqlens = []

    def createtextdata(self, size):
        for _ in range(size):
            rand_seq_len = np.random.choice(range(3, 7))
            self.seqlens.append(rand_seq_len)
            rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
            rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

            # Padding
            if rand_seq_len < 6:
                rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
                rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))

            self.even_sentences.append(" ".join([self.digit_to_word_map[r]
                                                 for r in rand_odd_ints]))
            self.odd_sentences.append(" ".join([self.digit_to_word_map[r]
                                                for r in rand_even_ints]))

        data = self.even_sentences + self.odd_sentences
        self.seqlens *= 2

        return data, self.seqlens

data, seqlens = Generate.createtextdata(10000)
print(seqlens[:6])
[5, 3, 4, 4, 3, 6]
```

Why keep the original sentence lengths? By _zero-padding_, we solved one technical problem but created another: if we 
naively pass these padded sentences through our RNN model as they are, it will process useless PAD symbols. This would 
both harm model correctness by processing “noise” and increase computation time. We resolve this issue by first storing 
the original lengths in the _seqlens_ array and then telling TensorFlow’s _tf.nn.dynamic_rnn()_ where each sentence ends.

We now map words to indices—word _identifiers_—by simply creating a dictionary with words as keys and indices as values. 
We also create the inverse map. Note that there is no correspondence between the word _IDs_ and the digits each word 
represents —the IDs carry no semantic meaning, just as in any NLP application with real data:

```python
def wordtoindex(self):
    """
    Words to indices—word identifiers—by simply creating a dictionary with
    words as keys and indices as values.
    :return: list of index to words map
    """
    for sent in self.data:
        for word in sent.lower.split():
            if word not in self.word2index_map:
                self.word2index_map[word] = self.index
                self.index += 1

def indextoword(self):
    """
    Map index to words
    :return:
    """
    index2word_map = {index: word for word, index in self.word2index_map.items()}
    vocabulary_size = len(index2word_map)
    return index2word_map, vocabulary_size
```

#### One Hot Encoding
After one hot encoding we will split the training and test set. All this work is done by functions present in datagenrator.py.

```python
gendata = Generate(10000)
gendata.createtextdata()
gendata.createlabels()
splitteddataset = gendata.splitdataset()
trainX, trainY, testX, testY, trainSeq, testSeq = splitteddataset
```

#### Create Placeholders
```python
import tensorflow as tf
# Create placeholders for dataset
_inputs = tf.placeholder(tf.int32, shape=[BATCHSIZE, TIMESTEPS])
_labels = tf.placeholder(tf.float32, shape=[BATCHSIZE, NUMCLASSES])

# sequencelens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[BATCHSIZE])
```

### Supervised Word Embeddings
Our text data is now encoded as lists of word _IDs_—each sentence is a sequence of integers corresponding to words. This 
type of atomic representation, where each word is represented with an ID, is not scalable for training deep learning models 
with large vocabularies that occur in real problems. We could end up with millions of such word IDs, each encoded in 
one-hot (binary) categorical form, leading to great data sparsity and computational issues. 

A powerful approach to work around this issue is to use word embeddings. The embedding is, in a nutshell, simply a mapping 
from high-dimensional one-hot vectors encoding words to lower-dimensional dense vectors. So, for example, if our vocabulary 
has size 100,000, each word in one-hot representation would be of the same size. The corresponding word vector—or word 
embedding—would be of size 300, say. The high-dimensional one-hot vectors are thus “embedded” into a continuous vector 
space with a much lower dimensionality.

It is helpful to think of word embeddings as basic hash tables or lookup tables, mapping words to their dense vector values. 
These vectors are optimized as part of the training process. Previously, we gave each word an integer index, and sentences 
are then represented as sequences of these indices. Now, to obtain a word’s vector, we use the built-in _tf.nn.embedding_lookup()_ 
function, which efficiently retrieves the vectors for each word in a given sequence of word indices:

```python
with tf.name_scope('embeddings'):
    embeddings = tf.Variable(tf.random_normal([len(gendata.index2word_map),
                                               EMBEDDINGDIMENSION], -1.0, 1.0),
                             name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
```


