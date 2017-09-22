from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):
    def __init__(self):
        # Data format
        self.image_height = 32
        self.image_width = 32
        self.image_channel = 1

        self.batch_size = 32

        # maximum of input loop length
        # We should pad loopsites to this dim
        self.max_loop_size = 12

        self.feature_dims = 256

        # vocab_size
        self.num_sites = self.image_height*self.image_width*self.image_channel
        # ice states are stored in 1D
        self.image_shape = [self.num_sites]

        # ConvNets (Too heavy now)
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.conv3_filters = 64
        self.conv1_kernel = 3
        self.conv2_kernel = 5
        self.conv3_kernel = 5
        self.fc1_hiddens = 128
        self.fc2_hiddens = 256
        # output features dimension should fit embedding layers' input size

        # LSTM
        self.initializer_scale = 0.08 # replaced by sqrt(3)
        self.embedding_size = self.feature_dims
        self.num_lstm_units = 64
        #self.num_lstm_layers = 3  # optional now
        self.lstm_dropout_keep_prob = 0.7

        # Solvers!
        self.learning_rate = 0.005
        self.clip_gradients = 5.0
