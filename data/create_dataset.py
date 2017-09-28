from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py as hf
import sys, os
import argparse

from data_utils import *
from loopalgo import *

# Global variables
L=32
parser = argparse.ArgumentParser(description="Create ice2loop dataset")
parser.add_argument("--state_dir", dest="state_dir", default="squareice_states_5000x1024.h5", type=str, help="Path to ice states dataset")
parser.add_argument("--loops_dir", dest="loops_dir", default="loops", type=str, help="Path to loops folder")
parser.add_argument('--num_states', dest='num_states', default=10, type=int, help='Number of starting states')
parser.add_argument('--max_looplen', dest='max_looplen', default=20, type=int, help='Assign maximum allowable loop length')
parser.add_argument('--max_loopnum', dest='max_loopnum', default=1, type=int, help='Max number of loops in single state')
parser.add_argument("--out", dest="out_name", default="IceLoop.h5", type=str, help="Output name of datset")
FLAGS = parser.parse_args()

if FLAGS.max_loopnum > 1:
    CHECK_LOOP=True
else:
    CHECK_LOOP=False

def read_icestates(path):
    ds = hf.File(path, 'r')
    ices = ds['icestates'][:]
    print ('Read states from {}, get data shape {}'.format(path, ices.shape))
    ds.close()
    return ices

def read_loops(path, num_states=10):
    loops = []
    for idx in range(num_states):
        fs = get_filelist(source_idx=idx, prefix='loopsites')
        loop_list = read_filelist(fs)
        num_of_loops = len(loop_list)
        loops.append(loop_list)
        print ('{} loops in index {}'.format(num_of_loops, idx))
    return loops

ices = read_icestates(FLAGS.state_dir)
loop_indexed_by_ices = read_loops(FLAGS.loops_dir, num_states=FLAGS.num_states)

STATES = []
LOOPS  = []

for ice_idx, loops in enumerate(loop_indexed_by_ices):
    # Plug in the pure ice state
    state = ices[ice_idx]
    for loop in loops:
        denoted_heads = [loop[0]]

        # Skip loop exceeds max length
        for nloop in range(FLAGS.max_loopnum-1):
            loop2 = np.random.choice(loops)
            denoted_heads.append(loop2[0])
            loop = np.concatenate((loop, loop2), axis=0)

        current_looplen = len(loop)
        if (current_looplen > FLAGS.max_looplen):
            continue
        padding_length = FLAGS.max_looplen - current_looplen

        # Check the loop satisfy the metropolis judgement
        if CHECK_LOOP:
            ok = pseudo_metropolis(state, loop, 32, 'loopsites')
            if (not ok):
                print ('loop is not accepatble')
                continue

        # Shift the sites number and add ending token
        padded_loop = [site+1 for site in loop] + padding_length * [0]
        #print (padded_loop)
        # Flipping corresponding state
        denoted_state = np.copy(state)
        denoted_state[denoted_heads] = 0 # or flip?
        #print ('denoted sites {}'.format(np.where(denoted_state == 0)[0]))
        # Append to dataset
        STATES.append(denoted_state)
        LOOPS.append(padded_loop)
    print ('Done index {}, now datset size is {}'.format(ice_idx, len(LOOPS)))

## TODO: We need to merge pure icestates
data_num = len(LOOPS)
if data_num < len(ices):
    for ice in ices[:data_num]:
        STATES.append(ice)
        LOOPS.append([0 for _ in range(FLAGS.max_looplen)])
else:
    for ice in ices:
        STATES.append(ice)
        LOOPS.append([0 for _ in range(FLAGS.max_looplen)])

dataset = hf.File(FLAGS.out_name, 'w')
dataset['STATES'] = STATES
dataset['LOOPS'] = LOOPS
dataset.close()
print ('{} dataset ({}) is created.'.format(FLAGS.out_name, len(LOOPS)))