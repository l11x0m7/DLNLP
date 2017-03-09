import glob
import cPickle as pickle
import random
import numpy as np
import os.path as op

SAVE_PARAMS_EVERY = 1000

def save_params(iter, params):
    with open('saved_params_{0}.npy'.format(iter)) as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate, f)

def load_saved_params():
    st = 0
    for f in glob.glob('saved_params_*.npy'):
        iter = int(op.splitext(op.basename(f))[0].split('_')[2])
        if iter>st:
            st = iter

    if st>0:
        with open('saved_params_%d' % st) as f:
            params = pickle.load(f)
            states = pickle.loads(f)
            return st, params, states
    else:
        return st, None, None

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    # Implement the stochastic gradient descent method in this
    # function.

    # Inputs:
    # - f: the function to optimize, it should take a single
    #     argument and yield two outputs, a cost and the gradient
    #     with respect to the arguments
    # - x0: the initial point to start SGD from
    # - step: the step size for SGD
    # - iterations: total iterations to run SGD for
    # - postprocessing: postprocessing function for the parameters
    #     if necessary. In the case of word2vec we will need to
    #     normalize the word vectors to have unit length.
    # - PRINT_EVERY: specifies every how many iterations to output

    # Output:
    # - x: the parameter value after SGD finishes

    # Anneal learning rate every several iterations

    ANNEAL_EVERY = 20000

    if useSaved:
        curiter, curx, state = load_saved_params()
        if curiter>0:
            start_iter = curiter
            x0 = curx
            step *= 0.5**(curiter/ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x:x

    expcost = 0

    for iter in range(start_iter+1, iterations+1):
        cost, grad = f(x)
        x -= step*grad
        x = postprocessing(x)

        if iter%ANNEAL_EVERY==0:
            step *= 0.5

        if iter%PRINT_EVERY==0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95*expcost + .05*cost
            print 'iter %d: %f' % (iter, expcost)

        if iter%SAVE_PARAMS_EVERY==0 and useSaved:
            save_params(iter, x)

    return x


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print "test 1 result:", t1
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print "test 2 result:", t2
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print "test 3 result:", t3
    assert abs(t3) <= 1e-6

    print ""


if __name__ == '__main__':
    sanity_check()