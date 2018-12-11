#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
from edward.models import Normal

def build_toy_dataset(N, w, noise_std=0.1):
    D = len(w)
    x = np.random.randn(N, D).astype(np.float32)
    y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
    return x, y

N = 40  # number of data points
D = 10  # number of features

w_true = np.random.randn(D)
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

#
# Build computation graph of model
#
X = tf.placeholder(tf.float32, [N, D])
w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))  # weights
b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))  # intercept
y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))  # Noisy response

#
# Create mean-field variational distribution
#
qw = Normal(mu=tf.Variable(tf.random_normal([D])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu=tf.Variable(tf.random_normal([1])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

#
# Infer by minimising KL(q||p)
#
inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run()

#
# Criticism
#
y_post = Normal(mu=ed.dot(X, qw) + qb, sigma=tf.ones(N))
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))

#
# Compare inferred weights to actual
#
plt.scatter(w_true, qw.mean().eval())
plt.savefig('weights.pdf')
qb.mean().eval()
