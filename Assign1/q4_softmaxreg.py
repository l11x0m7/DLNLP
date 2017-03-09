import random
import numpy as np


def getSentenceFeature(tokens, wordVectors, sentence):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # - tokens: a dictionary that maps words to their indices in
    #          the word vector list
    # - wordVectors: word vectors (each row) for all tokens
    # - sentence: a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence