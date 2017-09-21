from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ice2loop_model import IceToLoopModel
from data.data_utils import DataReader
from data.loopalgo import *
import configurations

tf.app.flags.DEFINE_string('ckpt_dir', 'logs/ice2loop', 'Path to the checkpoint directory')

FLAGS = tf.app.flags.FLAGS


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


def inference():
    config = configurations.ModelConfig()

    data = DataReader('data/MarkovSet.h5')


    with tf.Session() as sess:
        model = load_model(sess, config)

        num_samples = data.num_samples
        print ('Number of sample data: {}'.format(num_samples))

        ices, [input_seq], [target_seq], [mask_seq] = data.next_batch(1)
        initial_state = model.feed_image(sess, ices)

        print (initial_state.shape)
        print (input_seq)
        print (input_seq.shape)
        state = initial_state
        site = input_seq[0]

        ground_loop = []
        loop_sites = []
        confidence_loops = []
        ground_loop.append(site)
        confidence_loops.append(1)
        ground_loop.extend(target_seq)

        print ('Ground loop: {}'.format(ground_loop))
        loop_sites.append(site)

        print (mask_seq)
        max_steps = np.sum(mask_seq)

        for step in range(max_steps):
            softmax, new_state, _ = model.inference_step(sess,
                                                        input_feed=[site],
                                                        state_feed=state)
            new_state = initial_state
            site = np.argmax(softmax)
            prob = np.max(softmax)
            print ('Target Site: {}, Predict Site: {} with p={}'.format(target_seq[step], site, prob))
            if site == 0:
                break
            else:
                loop_sites.append(site)
                confidence_loops.append(prob)
        
        ground_truth = np.zeros(1024)
        predict = np.zeros(1024)
        confidence_map = np.zeros(1024)

        for s in ground_loop:
            if s != 0:
                ground_truth[s] = 1

        for s, p in zip(loop_sites, confidence_loops):
            if s != 0:
                predict[s] = 1 
                confidence_map[s] = p

        is_accept = pseudo_metropolis(ices[0], ground_loop, 32, prefix='loopsites')
        if is_accept:
            print ('Ground truth is Accept, {}'.format(ground_loop))
        is_accept = pseudo_metropolis(ices[0], loop_sites, 32, prefix='loopsites')
        if is_accept:
            print ('Generated Loop is Accept, {}'.format(loop_sites))
        else:
            print ('Generated Loop is Reject! {}'.format(loop_sites))

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(ground_truth.reshape(32,32), interpolation='None', cmap='plasma')
        ax1.set_xlabel('Ground Truth')

        ax2 = fig.add_subplot(122)
        ax2.imshow(predict.reshape(32,32), interpolation='None', cmap='plasma')
        ax2.set_xlabel('Prediction')

        '''
        ax3 = fig.add_subplot(133)
        ax3.imshow(confidence_map.reshape(32,32), interpolation='None', cmap='plasma')
        ax3.set_xlabel('Confidence')
        cbar = fig.colorbar(ax3, ticks=[-1, 0, 1], orientation='vertical')
        '''
        #cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

        plt.tight_layout()
        plt.savefig('ice2loop.png')
        plt.show()

        plt.imshow(confidence_map.reshape(32,32), interpolation='None', cmap='plasma')
        plt.colorbar()
        plt.savefig('confidence.png')
        plt.show()


def main():
    inference()

if __name__ == '__main__':
    main()