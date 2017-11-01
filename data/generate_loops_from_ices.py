from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import random
import h5py as hf
import sys, os
from loopalgo import *

L = 32
SAVEFIG = True
iceset = hf.File('squareice_states_5000x1024.h5', 'r')
ices = iceset['icestates']

if not os.path.exists('loops'):
    os.makedirs('loops')

if not os.path.exists('limgs') and SAVEFIG:
    os.makedirs('limgs')

## multiprocessing is needed
for idx1, s1 in enumerate(ices):
    for idx2, s2 in enumerate(ices):
        if (idx2 == idx1):
            # or idx2 <= idx1
            continue
        print ('State {} transits to state {}'.format(idx1, idx2))
        s1 = ices[idx1]
        s2 = ices[idx2]
        trans = s1-s2
        d1loops = trans_subset(s1, trans, L, from_idx=idx1, to_idx=idx2, dilation_times=1, save_img=SAVEFIG)
        d2loops = trans_subset(s1, trans, L, from_idx=idx1, to_idx=idx2, dilation_times=2, save_img=SAVEFIG)
        loops = d1loops + d2loops
        print (' capture {} loops from {} to {}'.format(len(loops), idx1, idx2))
        if (len(loops) is not 0):
            print ('Save loops information')
            np.save('loops/loopstate_{}-{}'.format(idx1, idx2), loops)
            loopsites = [np.nonzero(loop) for loop in loops]
            np.save('loops/loopsites_{}-{}'.format(idx1, idx2), loopsites)