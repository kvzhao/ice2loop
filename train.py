from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import tensorflow as tf

from ice2loop_model import IceToLoopModel
from data.data_utils import DataReader
import configurations

tf.app.flags.DEFINE_string('logdir', 'logs', 'Dir of training logs')
tf.app.flags.DEFINE_string('task_name', 'ice2loop', 'Name of this training task')

tf.app.flags.DEFINE_integer('num_steps', 50000, 'Number of training steps')
tf.app.flags.DEFINE_integer('save_freq', 10000, 'Save model after # of steps')
tf.app.flags.DEFINE_integer('eval_freq', 100, 'Save to summary after # of steps')

FLAGS = tf.app.flags.FLAGS

# first time using this
tf.logging.set_verbosity(tf.logging.INFO)

def main():
    model_config = configurations.ModelConfig()

    # Create directory
    logdir = FLAGS.logdir
    if not tf.gfile.IsDirectory(logdir):
        tf.logging.info('Create training directory: %s' % logdir)
        tf.gfile.MakeDirs(logdir)
    train_dir = '/'.join([logdir, FLAGS.task_name])
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info('Create the training task: %s' % train_dir)
        tf.gfile.MakeDirs(train_dir)
    
    # Build Graph
    g = tf.Graph()
    with g.as_default():
        model = IceToLoopModel(model_config, 'train')
        model.build_all()
        # Set up training rate (TODO)
        # Set up training operations
        model.init_optimizer()

        batch_size = model_config.batch_size

        logwriter = tf.summary.FileWriter(train_dir, graph=g)
    
    # Set up the Saver and Restoration
        with tf.Session() as sess:
            # Data Reader
            data = DataReader('data/MarkovSet.h5')
            num_samples = data.num_samples

            # Initailize the network
            sess.run(tf.global_variables_initializer())

            # Run training.
            for step in range(FLAGS.num_steps):
                image_batch, input_batch, target_batch, mask_batch = data.next_batch(batch_size)
                step_loss, summary = model.train_step(sess, image_batch, input_batch, target_batch, mask_batch)
                if step % FLAGS.eval_freq == 0:
                    print ('Step[ {} ]: step loss = {}'.format(step, step_loss))
                    logwriter.add_summary(summary, step)
                if step % FLAGS.save_freq == 0:
                    checkpoint_path = os.path.join(train_dir, 'Ice2Loop')
                    model.save(sess, checkpoint_path)

if __name__ == '__main__':
    main()
