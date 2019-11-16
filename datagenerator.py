"""
Generate sequence of text data for RNN.
We create sentences. We sample random digits and map them to the corresponding
“words” (e.g., 1 is mapped to “One,” 7 to “Seven,” etc.). Text sequences typically
have variable lengths, which is of course the case for all real natural language
data. To make our simulated sentences have different lengths, we sample for each
sentence a random length between 3 and 6 with np.random.choice(range(3, 7))—the
lower bound is inclusive, and the upper bound is exclusive.

Now, to put all our input sentences in one tensor (per batch of data instances),
we need them to somehow be of the same size—so we pad sentences with a length shorter
than 6 with zeros (or PAD symbols) to make all sentences equally sized (artificially).
This pre-processing step is known as **zero-padding**.
"""
import numpy as np

class Generate:
    """
    Class to create object of text data
    """
    def __init__(self, dataset_size):
        self.seqlen = 0
        self.dataset_size = dataset_size
        self.digit_to_word_map = {0: "PAD", 1: "One", 2: "Two", 3: "Three",
                                  4: "Four", 5: "Five", 6: "Six",
                                  7: "Seven", 8: "Eight", 9: "Nine"}
        self.even_sentences = []
        self.odd_sentences = []
        self.seqlens = []
        self.data = []
        self.word2index_map = {}
        self.index2word_map = {}
        self.index = 0
        self.labels = [1] * dataset_size + [0] * dataset_size

    def createtextdata(self):
        """
        Creates odd and even sentences.
        :param size: length of list of sentences
        :return:
            data: list of sentences
            seqlen: size of list
        """
        for _ in range(self.dataset_size):
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

        self.data = self.even_sentences + self.odd_sentences
        self.seqlens *= 2

    def createlabels(self):
        """
        Create labels which will have only 2 values(0,1).
        One-hot encode the labels
        :param seqlen: Length of the dataset
        :return: List of one-hot encoded labels
        """
        for i, label in enumerate(self.labels):
            one_hot_encoding = [0] * 2
            one_hot_encoding[label] = 1
            self.labels[i] = one_hot_encoding

    def splitdataset(self):
        """
        Split the dataset in train and test set
        :param data: complete dataset of sentences with total sentence of 20000
        :param labels: one hot encoded target values
        :param seqlen: List of length each sentences from dataset
        """
        data_indices = list(range(len(self.data)))
        np.random.shuffle(data_indices)

        dataset = np.array(self.data)[data_indices]
        labels = np.array(self.labels)[data_indices]
        seqlens = np.array(self.seqlens)[data_indices]

        train_X = dataset[:10000]
        test_X = dataset[10000:]

        train_Y = labels[:10000]
        test_Y = labels[10000:]

        train_Seq = seqlens[:10000]
        test_Seq = seqlens[10000:]

        return train_X, train_Y, test_X, test_Y, train_Seq, test_Seq

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
        self.index2word_map = {index: word for word, index in self.word2index_map.items()}
