#!/usr/bin/env python
"""
Fit data from a mixture of Gaussians using Edward.
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Seed
rd.seed(301)

#
# Generate data
#
N = 1000
p0 = .333333
k = rd.choice([0,1], size=N, p=[p0, 1-p0])
# Parameters for first component
mu0 = [3, -3]
sigma0 = np.diag(np.ones(2))
# Parameters for second component
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
mu1 = [0, 0]
sigma1 = np.diag([1, 9]) * np.matrix([[c, -s], [s, c]])
# Sample data
x = np.matrix([
    rd.multivariate_normal(mu0 if comp == 0 else mu1, sigma0 if comp == 0 else sigma1)
    for comp in k
    ])
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')
plt.style.use('seaborn-pastel')
plt.style.use('bmh')
plt.figure()
plt.scatter(x[:,0], x[:,1], c=k)
plt.savefig('figures/data.png')
