'''Variational Bayes for multinomial mixture model'''

import numpy as np
from util import *

def mmm_vb_fit(X, T, alpha=None, beta=None, pz=None, eps=0.00001, maxiter=1000):
    '''Fit a multinomial mixture model using variational Bayes.

       Parameters
       ----------
       X : 2D array of float
           Training data, an instance on each row.
       T : int
           Number of mixture components.
       alpha : 1D array of float
           Dirichlet prior for mixing proportions. If None, generate the prior
           randomly.
       beta : 2D array of float
           Dirichlet priors for component distributions, one component on each
           row. If None, generate priors randomly.
       pz : 2D array of float
           Initial component probabilities P(z|x), one instance x on each row,
           one component z on each column. If None, generate the initial
           probabilities randomly.
       eps : float
           Stop when change in P(z|x) is less than eps.
       maxiter : int
           Maximum number of iterations.

       Returns
       -------
       alpha : 1D array of float
           Dirichlet posterior for mixing proportions
       beta : 2D array of float
           Dirichlet posteriors for component distributions, one component on
           each row.
       pz : 2D array of float
           P(z|x), one instance x on each row, one component z on each column.
    '''
    V = X.shape[1]

    if pz is None:
        pz = np.random.dirichlet(np.ones(T), D)
    if alpha is None:
        alpha = np.random.rand(T)
    else:
        alpha = alpha.copy()
    if beta is None:
        beta = np.random.rand(T, V)
    else:
        beta = beta.copy()

    i = 0
    pz_diff = np.finfo('float').max
    while i < maxiter and pz_diff > eps:
        eth = dir_log_expect(alpha)
        eph = dir_log_expect(beta)

        alpha += np.sum(pz, axis=0)
        beta += pz.T.dot(X)

        c = X.dot(eph.T) + eth

        pz_new = np.exp(c)
        pz_new = (pz_new.T / np.sum(pz_new, axis=1)).T
        pz_diff = np.max(np.abs(pz_new-pz))
        pz = pz_new
        i += 1

    return alpha, beta, pz

