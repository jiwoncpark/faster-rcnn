import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.slim as slim
import layers as L

def build(input_tensor, num_classes=2, is_training=True, padding='SAME', reuse=None, keep_prob=0.5):
    with tf.variable_scope('net', 'image_to_head', reuse=reuse):

        with tf.variable_scope('net', 'image_to_head','conv1_1', reuse=reuse):
            net = L.conv2d(input_tensor=input_tensor, name='conv1_1', kernel=(3,3), stride=(2,2), num_filter=64, activation_fn=tf.nn.relu)
        
        with tf.variable_scope('net', 'image_to_head', 'pool1',reuse=reuse):
            net = L.max_pool (input_tensor=net, name="pool1",   kernel=(3,3), stride=(2,2))
    
        with tf.variable_scope('net', 'image_to_head', 'conv2_1',reuse=reuse):
            net = L.conv2d(input_tensor=net, name='conv2_1', kernel=(3,3), stride=(2,2), num_filter=128, activation_fn=tf.nn.relu)

        with tf.variable_scope('net', 'image_to_head', 'pool2',reuse=reuse):
            net = L.max_pool (input_tensor=net, name="pool2",   kernel=(3,3), stride=(2,2))
        '''
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], stride=2, padding=padding, scope='fc6')
        net = slim.dropout(net, keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        
        net = slim.dropout(net, keep_prob, is_training=is_training,
                           scope='dropout3')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc4')
        '''
        with tf.variable_scope('net', 'image_to_head', 'fc_layer',reuse=reuse):
            net = L.final_inner_product(input_tensor=net, name='fc_final', num_output=num_classes)
    return net
    #return net

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,256,256,1])
    net = build(x)
    print net.shape
