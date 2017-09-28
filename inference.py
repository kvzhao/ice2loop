from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ice2loop_model import IceToLoopModel
from data.inference_utils import InferenceData
from data.inference_utils import *
from data.loopalgo import *

tf.app.flags.DEFINE_string('ckpt_dir', 'logs/ice2loop', 'Path to the checkpoint directory')
tf.app.flags.DEFINE_string('test_data', 'data/squareice_states_10000x1024.h5', 'Path to testing dataset')
tf.app.flags.DEFINE_string('results', 'resutls', 'Folder name of expieriments results')
#tf.app.flags.DEFINE_string('eval_mode', 'demo', 'Modes of evaluation ()')

FLAGS = tf.app.flags.FLAGS

if not tf.gfile.IsDirectory(FLAGS.results):
    tf.gfile.MakeDirs(FLAGS.results)

def read_config(path):
    config_path = os.path.join(path, 'config.json')
    with open(config_path) as file:
        saved = json.load(file)
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Config(**saved)

def load_model(sess, config):
    model = IceToLoopModel(config, 'inference')
    model.build_all()
    if tf.train.checkpoint_exists(FLAGS.ckpt_dir):
        print ('Reloading model parameters...')
        model.restore(sess, FLAGS.ckpt_dir)
    else:
        raise ValueError(
            'No such file: {}'.format(FLAGS.ckpt_dir)
        )
    return model

def calculate_acceptance():
    # Calculate for all images
    pass


def single_image_analysis (sess, model, image, max_steps=20):
    # Create the folder for saving reuslts
    folderpath = os.path.join(FLAGS.results, 'single_image_update')
    if not tf.gfile.IsDirectory(folderpath):
        tf.gfile.MakeDirs(folderpath)

    # Create the denoted starting point
    #start_site = np.random.randint(1024)
    start_site = 200
    image[0, start_site] = 0
    print ('start point {}'.format(start_site))

    loops = []
    state = model.feed_image(sess, image)
    site = start_site

    for step in range(max_steps):
        softmax, new_state, _ = model.inference_step(sess, input_feed=[site], state_feed=state)
        state = new_state
        new_site = np.argmax(softmax)
        # probability map, maybe this information is useful
        prob = np.max(softmax)
        imshow_probmap(softmax, folderpath, 'probmap_{}'.format(step))
        print ('site {} with confid {}'.format(site, prob))
        site = new_site
        # shift loop site back    
        loops.append(site - 1)

    imshow_loop(loops, folderpath, 'genloop')

    return loops

def inference():
    config = read_config(FLAGS.ckpt_dir)
    data = InferenceData(FLAGS.test_data)

    with tf.Session() as sess:
        model = load_model(sess, config)
        ice = data.next_batch(1)

        loop = single_image_analysis(sess, model, ice)
        print (loop)


def main():
    inference()

if __name__ == '__main__':
    main()