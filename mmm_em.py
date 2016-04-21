'''EM algorithm for multinomial mixture model'''

import numpy as np
from scipy.misc import logsumexp

def mmm_em_fit(X, T, pz=None, eps=0.00001, maxiter=1000):
    '''Fit a multinomial mixture model using EM.

       Parameters
       ----------
       X : 2D array of float
           Training data, an instance on each row.
       T : int
           Number of mixture components.
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
       theta : 1D array of float
           Mixing proportions.
       phi : 2D array of float
           Component distributions, one component on each row.
       pz : 2D array of float
           P(z|x), an instance x on each row, a component z on each column.
    '''
    D = X.shape[0]
    Nd = np.sum(X, axis=1)
    if pz is None:
	    pz = np.random.dirichlet(np.ones(T), D)

    pz_diff = np.finfo('float').max
    i = 0
    while i < maxiter and pz_diff > eps:
        theta = np.sum(pz, axis=0) / D
        phi = (X.T.dot(pz) / np.sum(pz.T * Nd, axis=1)).T
        L = np.log(theta) + X.dot(np.log(phi.T))
        pz_new = np.exp((L.T - logsumexp(L, axis=1)).T)
        pz_diff = np.max(np.abs(pz_new-pz))
        pz = pz_new
        i += 1

    return theta, phi, pz

