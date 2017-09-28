import tensorflow as tf

class ConvNet(object):
    def __init__(self, config, mode):
        assert mode.lower() in ['train', 'inference']
        self.mode = mode.lower()
        self.config = config

        # hyper-params of ConvNet
        self.conv1_filters = config.conv1_filters
        self.conv1_kernel = config.conv1_kernel
        self.conv2_filters = config.conv2_filters
        self.conv2_kernel = config.conv2_kernel
        self.conv3_filters = config.conv3_filters
        self.conv3_kernel = config.conv3_kernel

        self.fc1_hiddens = config.fc1_hiddens
        self.fc2_hiddens = config.fc2_hiddens

        self.image_width = config.image_width
        self.image_height = config.image_height
        self.image_channel = config.image_channel
        self.input_img_shape = [self.image_width, self.image_height, self.image_channel]
    
    def build_model(self, inputs):
        print ('\tStart building model...')
        self._build_convet(inputs)

        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return self.features

        print ('\tDone.')
    
    def _init_placeholders(self):
        # image format: NHWC
        print ('\t\tInit placeholders')
        img_size = self.image_height * self.image_width * self.image_channel
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, img_size], name='img_inputs')
    
    def _build_convet(self, inputs):
        print ('\t\tInit ConvNet')
        shape = inputs.get_shape()[1:]
        if (shape == self.input_img_shape):
            print ('\t\tImage shape is same as inputs shape')
        else:
            print ('\t\tShape does not match, reshape from {} to {}'.format(shape, self.input_img_shape))
            inputs = tf.reshape(inputs, [-1, self.image_height, self.image_width, self.image_channel])
        conv1 = tf.layers.conv2d(inputs,
                            filters=self.conv1_filters,
                            kernel_size=[self.conv1_kernel, self.conv1_kernel], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            padding="same", activation=tf.nn.relu, name='Conv1')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='Pool1')

        conv2 = tf.layers.conv2d(pool1, 
                            filters=self.conv2_filters,
                            kernel_size=[self.conv2_kernel, self.conv2_kernel], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            padding="same", activation=tf.nn.relu, name='Conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='Pool2')
        '''

        conv3 = tf.layers.conv2d(pool2, 
                            filters=self.conv3_filters,
                            kernel_size=[self.conv3_kernel, self.conv3_kernel], 
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            padding="same", activation=tf.nn.relu, name='Conv3')
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='Pool3')
        '''

        flatten = tf.contrib.layers.flatten(pool2)

        fc1 = tf.layers.dense(flatten, self.fc1_hiddens, activation=tf.nn.relu,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='FC1')
        if self.mode.lower() == 'train':
            fc1 = tf.layers.dropout(fc1, rate=0.5 , name='FC1Drop')

        fc2 = tf.layers.dense(fc1, self.fc2_hiddens, activation=tf.nn.relu,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='FC2')

        if self.mode.lower() == 'train':
            fc2 = tf.layers.dropout(fc2, rate=0.5, name='FC2Drop')

        self.features = fc2

    def restore(self, sess, path):
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('model restore from {}'.format(checkpoint.model_checkpoint_path))
        else:
            raise ValueError('Can not load from checkpoints')
        