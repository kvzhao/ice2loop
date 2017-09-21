from __future__ import division
import numpy as np
from random import shuffle

import scipy
from scipy.ndimage.measurements import label
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rint = np.random.randint

def pbc(s, d, l):
    return ((s+d)%l + l) % l

def flip_along(state, loop):
    for l in loop:
        # NOTICE! TRICK HERE!
        if l != 0:
            state[l] *= -1

def apply_trans(state, trans):
    flip_along(state, np.nonzero(trans)[0])

def trans_subset(state, tmap, L, from_idx= 0, to_idx = 1, dilation_times=2, save_img=True):
    '''
        Retrieve loops from transition map with dilation algorithm
        state: original state
        tmap : transition map
        index: global index indicates the original state
        L    : system size
        dilation_times: default is 2

        return: all loops we can extract from state
    '''
    # preprocessing
    N = L ** 2
    tmap = tmap.reshape(L, L)
    tmap[tmap == +2] = +1
    tmap[tmap == -2] = -1
    empty_area = (tmap + 1) % 2

    # retrieve
    labeled_arr, num_features = label(empty_area)
    labeled_list = labeled_arr.flatten().tolist()
    labeled_set = set(labeled_list)
    labeled_count = [labeled_list.count(x) for x in labeled_set]

    labeled = dict(zip(labeled_set, labeled_count))
    # return
    loops = [] 

    for i, l in enumerate(sorted(labeled, key=labeled.get, reverse=True)[1:]):
        [x, y] = np.where(labeled_arr==l)
        z = np.zeros_like(tmap)
        z[x,y] = 1
        # two times dilation
        z_b = z
        for _ in range(dilation_times):
            z_b = binary_dilation(z_b)
        output = (z_b-z) * tmap

        ## we should check the quality of output
        proposed_trans = output.reshape(N)

        accept = pseudo_metropolis(state, proposed_trans, L)

        # save the state and loop if accepted
        if (accept):
            loops.append(proposed_trans)

        if save_img and accept:
            plt.imshow(output, 'gray', interpolation='None')
            plt.savefig('limgs/{}-{}_{}.png'.format(from_idx, to_idx, i))
        
    return loops

def cal_energy(state, L):
    eng = 0.0
    J = 1.0
    for site, spin in enumerate(state):
        neighbors = get_neighbor(site, L)
        se = np.sum(state[neighbors])
        eng += J * spin * se
    return eng / (2.0*L**2)

def pseudo_metropolis(state, loop, L, prefix='loopstates'):
    accpet = False
    s = np.copy(state)
    E0 = cal_energy(state, L)
    print ('Initial Energy {}'.format(E0))
    if prefix == 'loopstates':
        apply_trans(s, loop)
    elif prefix == 'loopsites':
        flip_along(s, loop)
    diff = state-s
    loop_size = int(np.sum(np.abs(diff))/2)
    Et = cal_energy(s, L)
    print ('Updated Energy {}'.format(Et))
    dE = Et - E0
    if (dE == 0.0):
        print ('Accept with loop size: {}'.format(loop_size))
        accpet = True
    else:
        print ('Reject with dE = {} and size {} loop'.format(dE, loop_size))
    return accpet

def transit(state, loop, L, prefix='loopsites'):
    accpet = False
    s = np.copy(state)
    E0 = cal_energy(state, L)
    if prefix == 'loopstates':
        apply_trans(s, loop)
    elif prefix == 'loopsites':
        flip_along(s, loop)
    loop_size = int(np.sum(np.abs(state-s))/2)
    Et = cal_energy(s, L)
    dE = Et - E0
    if (dE == 0.0):
        #print ('Accept with loop size: {}'.format(loop_size))
        accpet = True
    #else:
        #print ('Reject with dE = {} and size {} loop'.format(dE, loop_size))
    return accpet, s

def get_neighbor(site, L):
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

def long_loop(state, L):
    loop = []
    start_site = rint(L)
    print ('start site: {}'.format(start_site))
    print ('start spin: {}'.format(state[start_site]))
    current_site = None
    while (start_site is not current_site):
        if current_site is None:
            current_site = start_site
            #loop.append(current_site)
            # flip and create defect
            state[start_site] *= -1
        spin = state[current_site]
        neighbors = get_neighbor(current_site, L)
        print ('neighbors of {} are {}'.format(current_site, neighbors))
        neighborspins = state[neighbors]
        # choose neighbor with same spin and propagte
        candidates = [site for site in neighbors if state[site] == 1 * spin]
        print ('candidates: {} with spins = {}'.format(candidates, state[candidates]))
        # move the defect
        shuffle(candidates)
        current_site = candidates[0]
        if current_site is not start_site:
            loop.append(current_site)
    print(loop)
    return list(set(loop))