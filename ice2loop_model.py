from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np
import tensorflow as tf

from convnet import ConvNet

class IceToLoopModel(object):
    def __init__ (self, config, mode):
        assert mode in ['train', 'inference']
        self.config = config
        self.mode = mode

        # utilities
        sqrt3 = math.sqrt(3)
        self.initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

        self.num_sites = config.image_height * config.image_width * config.image_channel
        self.image_shape = [self.num_sites]

    def init_placeholders(self):
        if self.mode == 'inference':
            image_feed = tf.placeholder(dtype=tf.float32,
                            shape=[None, ] + self.image_shape,
                            name='image_feed')
            #self.images = image_feed
            self.images = tf.expand_dims(image_feed, 0)
            # What input_feed do?
            input_feed = tf.placeholder(dtype=tf.int32, 
                                        shape=[None, ], name='input_feed')
            self.input_sequences = tf.expand_dims(input_feed, 1)
            self.input_masks = None
            self.target_sequences = None
        else:
            self.images = tf.placeholder(dtype=tf.float32,
                            shape=[None, ] + self.image_shape,
                            name='images')
            self.input_sequences = tf.placeholder(dtype=tf.int32, 
                                        shape=[None, self.config.max_loop_size-1], name='input_sequences')
            self.target_sequences= tf.placeholder(dtype=tf.int32, 
                                        shape=[None, self.config.max_loop_size-1], name='target_sequences')
            self.input_masks = tf.placeholder(dtype=tf.int32, 
                                        shape=[None, self.config.max_loop_size-1], name='input_masks')
            #Q: why we should assign shape here?

    def build_image_embedding(self):
        print ('Build image embedding')
        with tf.variable_scope('ConvNet') as scope:
            convnet = ConvNet(self.config, self.mode)
            cnn_output = convnet.build_model(self.images)
        with tf.variable_scope('IcestateEmbedding') as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=cnn_output,
                num_outputs=self.config.feature_dims,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        tf.constant(self.config.embedding_size, name="embedding_size")
        self.image_embeddings = image_embeddings

    def build_sequence_embedding(self):
        print ('Build sequence embedding')
        with tf.variable_scope('LoopSiteEmbedding'), tf.device('/cpu:0'):
            embedding_map = tf.get_variable(
                name='map',
                shape=[self.num_sites, self.config.embedding_size],
                initializer=self.initializer)
            sequence_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_sequences)
            self.sequence_embeddings = sequence_embeddings
            print ('Shape of sequence embedding')
            print (self.sequence_embeddings)
            print (self.sequence_embeddings.get_shape())

    def build_model(self):
        print ('Build Ice2Loop Model')
        # TODO: handle multiple layers
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units, state_is_tuple=True)
        if self.mode == 'train':
            # Add dropout layer
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)
        with tf.variable_scope('SequenceDecoder', initializer=self.initializer) as lstm_scope:
            # Feed the image embeddings to set the initial LSTM state
            print (self.image_embeddings.get_shape())
            zero_state = lstm_cell.zero_state(
                batch_size=tf.shape(self.image_embeddings)[0], dtype=tf.float32)
                # need more care about
                #batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
                # Note: We can use tf.shape(x) as argument rather than x.get_shape()
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)
            
            lstm_scope.reuse_variables()

            if self.mode == 'inference':
                # In inference mode, use concatenated states for convenient feeding and
                # fetching.
                tf.concat(axis=1, values=initial_state, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step.
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.sequence_embeddings, axis=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.concat(axis=1, values=state_tuple, name="state")
            else:
                sequence_length = tf.reduce_sum(self.input_masks, axis=1)
                print (sequence_length)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.sequence_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)
        # stack batches vertically
        # TODO: catch this up
        print ('lstm_output')
        print (lstm_outputs)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        print ('lstm_output')
        print (lstm_outputs)

        with tf.variable_scope('Logits') as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.num_sites,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)
            
        print ('logits')
        print (logits)

        if self.mode == 'inference':
            tf.nn.softmax(logits, name='softmax')
        else:
            targets = tf.reshape(self.target_sequences, [-1])
            weights = tf.to_float(tf.reshape(self.input_masks, [-1]))
            print (targets)
            print (weights)

            # Compute losses
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            print (losses)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                tf.reduce_sum(weights),
                                name='batch_loss')
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()
            
            tf.summary.scalar('losses/batch_loss', batch_loss)
            tf.summary.scalar('losses/totoal_loss', total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram('parameters/' + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.
    
    def setup_global_step(self):
        print ('Set global step')
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step',
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build_all(self):
        print ('Start building the whole system...')
        self.init_placeholders()
        self.build_image_embedding()
        self.build_sequence_embedding()
        self.build_model()
        self.setup_global_step()

        self.summary_op = tf.summary.merge_all()
        print ('Done.')
    
    def init_optimizer(self):
        if self.mode == 'train':
            print ('Initialize optimizer')
            trainable_params = tf.trainable_variables()
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            gradients = tf.gradients(self.total_loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip_gradients)
            self.updates = self.optimizer.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

    
    def train_step(self, sess, images_batch, input_batch, target_batch, mask_batch):
        input_feed = {
            self.images: images_batch,
            self.input_sequences: input_batch,
            self.target_sequences: target_batch,
            self.input_masks: mask_batch
        }
        output_feed = [self.updates, self.total_loss, self.summary_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2] # return the loss

    def inference_step(self, sess, input_feed, state_feed):
        softmax, state_output = sess.run(
            fetches=['softmax:0', 'SequenceDecoder/state:0'],
            feed_dict={
                "input_feed:0": input_feed,
                "SequenceDecoder/state_feed:0": state_feed 
            })
        return softmax, state_output, None # Why None?

    def feed_image(self, sess, input_image):
        initial_state = sess.run(fetches='SequenceDecoder/initial_state:0', 
                                    feed_dict={'image_feed:0': input_image})
        return initial_state

    def save(self, sess, path, var_list=None):
        saver = tf.train.Saver()
        save_path = saver.save(sess, path, self.global_step)
        print ('model (steps: {}) saved at {}'.format(self.global_step, path))
    
    def restore(self, sess, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        model_path = checkpoint.model_checkpoint_path
        if checkpoint and model_path:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print ('model restored from {}'.format(model_path))
        else:
            raise ValueError('Can not load from checkpoints!')