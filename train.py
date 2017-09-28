from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import json
import numpy as np
import tensorflow as tf

from ice2loop_model import IceToLoopModel
from data.data_utils import DataReader

# Folder & Path
tf.app.flags.DEFINE_string('logdir', 'logs', 'Dir of training logs')
tf.app.flags.DEFINE_string('task_name', 'ice2loop', 'Name of this training task')
tf.app.flags.DEFINE_string('data_path', 'data/IceLoop.h5', 'The path of training dataset')

# Data Information & format
tf.app.flags.DEFINE_integer('image_height', 32, 'Size of input configuration')
tf.app.flags.DEFINE_integer('image_width', 32, 'Size of input configuration')
tf.app.flags.DEFINE_integer('image_channel', 1, 'Depth of input configuration')
tf.app.flags.DEFINE_integer('num_sites', 32 * 32 * 1, 'Number of sites in input configuration')
tf.app.flags.DEFINE_integer('max_loop_size', 20, 'Maximum of input sequence length, padding when exceeded')

# Architectures: CNN & LSTM
tf.app.flags.DEFINE_integer('conv1_filters', 32, 'Number of convolutional filters')
tf.app.flags.DEFINE_integer('conv2_filters', 64, 'Number of convolutional filters')
tf.app.flags.DEFINE_integer('conv3_filters', 64, 'Number of convolutional filters')
tf.app.flags.DEFINE_integer('conv1_kernel', 3, 'Convolutional kernel size')
tf.app.flags.DEFINE_integer('conv2_kernel', 5, 'Convolutional kernel size')
tf.app.flags.DEFINE_integer('conv3_kernel', 5, 'Convolutional kernel size')
tf.app.flags.DEFINE_integer('fc1_hiddens', 128, 'Number of hidden units of fully connected layer')
tf.app.flags.DEFINE_integer('fc2_hiddens', 256, 'Number of hidden units of fully connected layer')

tf.app.flags.DEFINE_integer('feature_dims', 256, 'Dimension of encoded features')
# TODO: unify this two to encoding size
tf.app.flags.DEFINE_integer('embedding_size', 256, 'Embedding size which equals to feature dims')

tf.app.flags.DEFINE_integer('num_lstm_units', 64, 'Number of LSTM hidden units')
tf.app.flags.DEFINE_float('lstm_dropout_keep_prob', 0.7, 'Dropout probability of LSTM Layers')

# Solver/ Optimizer
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_float('clip_gradients', 5.0, 'Clipping of gradient values')

# Training process
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_integer('num_steps', 50000, 'Number of training steps')
tf.app.flags.DEFINE_integer('save_freq', 10000, 'Save model after # of steps')
tf.app.flags.DEFINE_integer('eval_freq', 100, 'Save to summary after # of steps')

FLAGS = tf.app.flags.FLAGS

# first time using this
tf.logging.set_verbosity(tf.logging.INFO)

print ('Task: {} is launching... '.format(FLAGS.task_name))

def main():
    # Create directory
    logdir = FLAGS.logdir
    if not tf.gfile.IsDirectory(logdir):
        tf.logging.info('Create training directory: %s' % logdir)
        tf.gfile.MakeDirs(logdir)
    train_dir = '/'.join([logdir, FLAGS.task_name])
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info('Create the training task: %s' % train_dir)
        tf.gfile.MakeDirs(train_dir)
    else:
        # The model already exist
        print ('Existing model.')
    
    # Save configs
    config = dict(FLAGS.__flags.items())
    with open(os.path.join(train_dir, 'config.json'), 'w') as file:
        json.dump(config, file)
    print ('Save training configuration to {}.'.format(train_dir))

    # Build Graph
    g = tf.Graph()
    with g.as_default():
        model_config = FLAGS
        model = IceToLoopModel(model_config, 'train')

        # Build the model
        model.build_all()

        # Set up training operations
        model.init_optimizer()

        logwriter = tf.summary.FileWriter(train_dir, graph=g)
    
    # Set up the Saver and Restoration
        with tf.Session() as sess:
            # Data Reader
            data = DataReader(FLAGS.data_path)
            num_samples = data.num_samples
            batch_size = FLAGS.batch_size

            # Initailize the network parameters
            sess.run(tf.global_variables_initializer())

            # Run training.
            for step in range(FLAGS.num_steps):

                image_batch, input_batch, target_batch, mask_batch = data.next_batch(batch_size)

                step_loss, summary = model.train_step(sess, image_batch, input_batch, target_batch, mask_batch)

                # TODO save validation accuracy
                if step % FLAGS.eval_freq == 0:
                    print ('Step[ {} ]: step loss = {}'.format(step, step_loss))
                    logwriter.add_summary(summary, step)

                if step % FLAGS.save_freq == 0:
                    checkpoint_path = os.path.join(train_dir, 'Ice2Loop')
                    model.save(sess, checkpoint_path)

if __name__ == '__main__':
    main()
