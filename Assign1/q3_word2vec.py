import numpy as np
import random

from q1_softmax import softmax

from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    return x/np.sqrt(np.sum(x**2, axis=1)).reshape((x.shape[0], 1)) + 1e-30

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
    print "Test passed!"

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    probability = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probability[target])
    delta = probability
    delta[target] -= 1
    V = delta.shape[0]
    N = predicted.shape[0]
    gradPred = (delta.reshape((1, V)).dot(outputVectors)).flatten()
    grad = delta.reshape((V, 1)).dot(predicted.reshape((1, N)))

    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE

    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

    indices = [target]
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices.append(newidx)

    labels = np.array([1] + [-1]*K)
    vecs = outputVectors[indices,:]
    mid = sigmoid(labels*vecs.dot(predicted))
    cost = -np.sum(np.log(mid))

    delta = labels*(mid-1)
    gradPred = delta.reshape((1, K+1)).dot(vecs).flatten()
    gradtmp = predicted.reshape((predicted.shape[0], 1)).dot(delta.reshape((1, K+1))).T
    for i in xrange(K+1):
        grad[indices[i]] += gradtmp[i,:]

#     t = sigmoid(predicted.dot(outputVectors[target,:]))
#     cost = -np.log(t)
#     delta = t - 1

#     gradPred += delta * outputVectors[target, :]
#     grad[target, :] += delta * predicted

#     for k in xrange(K):
#         idx = dataset.sampleTokenIdx()

#         t = sigmoid(-predicted.dot(outputVectors[idx,:]))
#         cost += -np.log(t)
#         delta = 1 - t

#         gradPred += delta * outputVectors[idx, :]
#         grad[idx, :] += delta * predicted

    ### END YOUR CODE

    return cost, gradPred, grad

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    target = tokens[currentWord]
    N = inputVectors.shape[1]
    Vc = np.zeros((N, ))
    for word in contextWords:
        idx = tokens[word]
        Vc += inputVectors[idx,:]

    cost, gd, gradOut = word2vecCostAndGradient(Vc, target, outputVectors, dataset)
    gradIn = np.zeros(inputVectors.shape)
    for word in contextWords:
        idx = tokens[word]
        gradIn[idx,:] += gd

    ### END YOUR CODE

    return cost, gradIn, gradOut

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE

    currentI = tokens[currentWord]
    predicted = inputVectors[currentI,:]
    N = inputVectors.shape[1]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    Vc = inputVectors[tokens[currentWord],:]
    for word in contextWords:
        target = tokens[word]
        c, gd, gdp = word2vecCostAndGradient(Vc, target, outputVectors, dataset)
        cost += c
        gradIn[currentI,:] += gd
        gradOut += gdp

    ### END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    dataset = type("dummy", (), {})()
    def dummySampleTokenIdx():
        return np.random.randint(0, 4)

    def getRandomContext(C):
        tokens = ['a', 'b', 'c', 'd', 'e']
        return tokens[np.random.randint(0, 4)], \
               [tokens[np.random.randint(0, 4)] for i in xrange(2* C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(12345)
    np.random.seed(54321)

    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4)])


    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)



if __name__ == '__main__':
    test_word2vec()