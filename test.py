import tensorflow as tf
import numpy as np
import pdb
from math import sqrt, cos, sin, pi

S = 4
K = 2
N = 3
zdim = 5

# zero = tf.constant(0., shape=[S, zdim])
# one = tf.constant(1., shape=[S, zdim])
# two = tf.constant(2., shape=[S, zdim])
# stack = tf.stack([zero,one,two],axis=1)
# reshape = tf.reshape(tf.transpose(stack,perm=[1,0,2]),[-1,zdim])

# labels = tf.constant(np.random.randint(0,N,size=(S,),dtype=np.int32))
# recons = tf.constant(np.random.randint(0,N,size=(K,),dtype=np.int32))
# one_hot_labels = tf.one_hot(labels, N)
# y1 = tf.expand_dims(one_hot_labels,axis=1)
# y2 = tf.one_hot(recons, N)
# sqr_dif = tf.square(y1 - y2)
# c = tf.reduce_sum(sqr_dif, axis=-1)

# sample_x = tf.constant(np.arange(S*K*zdim,dtype=np.float32).reshape((S,K,zdim)))
# sample_y = tf.constant(np.arange(0,2*S*K*zdim,2,dtype=np.float32).reshape((S,K,zdim)))
# norms_x = tf.reduce_sum(tf.square(sample_x), axis=-1, keepdims=True)
# dotprod = tf.matmul(tf.transpose(sample_x,perm=[1,0,2]),
#                     tf.transpose(sample_x,perm=[1,2,0]))
# dotprod_x = tf.transpose(dotprod,perm=[1,0,2])
# distances_x = norms_x + tf.transpose(norms_x) - 2. * dotprod_x
#
# norms_y = tf.reduce_sum(tf.square(sample_y), axis=-1, keepdims=True)
# dotprod = tf.matmul(tf.transpose(sample_y,perm=[1,0,2]),
#                     tf.transpose(sample_y,perm=[1,2,0]))
# dotprod_y = tf.transpose(dotprod,perm=[1,0,2])
# distances_y = norms_y + tf.transpose(norms_y) - 2. * dotprod_y
# dotprod = tf.matmul(tf.transpose(sample_x,perm=[1,0,2]),
#                     tf.transpose(sample_y,perm=[1,2,0]))
# dotprod_xy = tf.transpose(dotprod,perm=[1,0,2])
# distances = norms_x + tf.transpose(norms_y) - 2. * dotprod_xy
# diag = tf.trace(tf.transpose(distances,perm=[1,0,2]))
# out = tf.reduce_sum(distances,axis=[0,-1]) - diag

axis = 1
logits = tf.log(tf.constant(np.arange(1,S*N*K+1,dtype=np.float32).reshape((S,N,K)))/S*N*K)
l_max = tf.reduce_max(logits, axis=axis, keepdims=True)
logsumexp_shifted = tf.reduce_sum(tf.exp(logits-l_max),axis=axis)
logsumexp =  l_max[:,0] + logsumexp_shifted



sess = tf.Session()
log = sess.run(logits)
lmax = sess.run(l_max)
lse_shift = sess.run(logsumexp_shifted)
lse = sess.run(logsumexp)

# input = sess.run(input)
# input_reshape = sess.run(input_reshape)
# l1 = sess.run(l1)
# output = sess.run(output)
# zero = sess.run(zero)
# one = sess.run(one)
# two = sess.run(two)
# stack = sess.run(stack)
# reshape = sess.run(reshape)
# y1 = sess.run(y1)
# y2 = sess.run(y2)
# sqr_dif = sess.run(sqr_dif)
# c = sess.run(c)
# idx_label = sess.run(idx_label)
# l_pi = sess.run(l_pi)
# x = sess.run(sample_x)
# nx = sess.run(norms_x)
# xx = sess.run(dotprod_x)
# dx = sess.run(distances_x)
# y = sess.run(sample_y)
# ny = sess.run(norms_y)
# yy = sess.run(dotprod_y)
# dy = sess.run(distances_y)
# dxy = sess.run(distances)
# diag = sess.run(diag)
# out = sess.run(out)
# rp = sess.run(res1_pz)
# rq = sess.run(res1_qz)
# r1 = sess.run(res1)
# r1_diag = sess.run(res1_diag)
# r1_corr = sess.run(res1_corr)
# r2 = sess.run(res2)
# r = sess.run(res)

pdb.set_trace()
