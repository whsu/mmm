import numpy as np
from scipy.special import psi

def dir_log_expect(alpha):
	if len(alpha.shape) == 1:
		return psi(alpha) - psi(np.sum(alpha))
	else:
		return (psi(alpha).T - psi(np.sum(alpha, axis=1))).T

