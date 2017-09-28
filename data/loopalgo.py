from __future__ import division
import numpy as np
from random import shuffle
import tensorflow as tf
from tensorflow.python.ops.script_ops import py_func

rint = np.random.randint

def cal_energy(state, L=32):
    eng = 0.0
    J = 1.0
    for site, spin in enumerate(state):
        neighbors = get_neighbor(site, L)
        se = np.sum(state[neighbors], dtype=np.float32)
        eng += J * spin * se
    eng = eng / (2.0*L**2)
    return np.float32(eng)

def flip_along(state, loop):
    for l in loop:
        # NOTICE! TRICK HERE!
        if l != 0:
            state[l] *= -1

def pseudo_metropolis(state, loop, L, prefix='loopsites'):
    accpet = False
    s = np.copy(state)
    E0 = cal_energy(state, L)
    #print ('Initial Energy {}'.format(E0))
    if prefix == 'loopstates':
        apply_trans(s, loop)
    elif prefix == 'loopsites':
        flip_along(s, loop)
    diff = state-s
    loop_size = int(np.sum(np.abs(diff))/2)
    Et = cal_energy(s, L)
    #print ('Updated Energy {}'.format(Et))
    dE = Et - E0
    if (dE == 0.0):
        #print ('Accept with loop size: {}'.format(loop_size))
        accpet = True
    #else:
        #print ('Reject with dE = {} and size {} loop'.format(dE, loop_size))
    return accpet

def get_neighbor(site, L=32):
    pbc = lambda s, d, l: ((s+d)%l + l) % l
    x, y = int(site%L), int(site/L)
    neighbors = []
    xp = pbc(x, +1, L)
    xm = pbc(x, -1, L)
    yp = pbc(y, +1, L)
    ym = pbc(y, -1, L)
    neighbors.append(xp + y  * L)
    neighbors.append(x  + ym * L)
    neighbors.append(xm + y  * L)
    neighbors.append(x  + yp * L)
    if (x+y) % 2 == 0:
        # even
        neighbors.append(xp + yp  * L)
        neighbors.append(xm + ym  * L)
    else:
        # odd
        neighbors.append(xm + yp  * L)
        neighbors.append(xp + ym  * L)
    return neighbors

# compute energy using tensorflow
inp = tf.placeholder(tf.float32, shape=[None,], name='inplaceholder')

tf_cal_energy = tf.py_func(cal_energy, [inp], tf.float32)
