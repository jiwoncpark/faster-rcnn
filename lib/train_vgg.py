import os,sys
import numpy as np
import tensorflow as tf
import vgg_net
from rcnn_train.toydata_generator import ToydataGenerator

if len(sys.argv)>1:
  load_file = sys.argv[1]
else:
  load_file = None


batch_size_tr = 20
batch_size_val = 20
train_io = ToydataGenerator(N=256, max_tracks=None,  max_kinks=3, dtheta=-1, batch_size=batch_size_tr, classification=True)
val_io = ToydataGenerator(N=256, max_tracks=None, max_kinks=3, dtheta=np.radians(1), batch_size=batch_size_val, classification=True)

sess = tf.InteractiveSession()

data_tensor    = tf.placeholder(tf.float32,  [None, 256*256],name='x')
data_tensor_2d = tf.reshape(data_tensor,[-1,256,256,1])
label_tensor   = tf.placeholder(tf.int64, [None],name='labels')
prediction_tensor = tf.placeholder(tf.int64, [None],name='labels')

#RESHAPE IMAGE IF NEED BE                                                     
#tf.summary.image('input',data_tensor_2d,10)

net = vgg_net.build(input_tensor=data_tensor_2d, num_classes=2)

with tf.name_scope('softmax'):
  softmax = tf.nn.softmax(logits=net)

with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=label_tensor))
  tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), label_tensor)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

saver= tf.train.Saver()

if load_file:
  print "reading in model variables..."
  vlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  reader = tf.train.Saver(var_list=vlist)
  reader.restore(sess, load_file)

sess.run(tf.global_variables_initializer())
merged_summary=tf.summary.merge_all()
writer=tf.summary.FileWriter(logdir='log')
writer.add_graph(sess.graph)

# training loop
save_iter = int(10)
for i in range(1000000):

    blob = train_io.fetch_batch()

    if save_iter and (i+1)%save_iter == 0:        
        s = sess.run(merged_summary, feed_dict={data_tensor:blob['data'].reshape(-1, 256*256), label_tensor:blob['class_labels']})
        writer.add_summary(s,i)

        train_accuracy = accuracy.eval(feed_dict={data_tensor:blob['data'].reshape(-1, 256*256), label_tensor: blob['class_labels']})
        print("step %d, training accuracy %g"%(i, train_accuracy))

        save_path = saver.save(sess,'model/ckpt',global_step=i)
        print 'saved @',save_path

    sess.run(train_step,feed_dict={data_tensor: blob['data'].reshape(-1, 256*256), label_tensor: blob['class_labels']})

    if i%20 ==0:
        blob_test = val_io.fetch_batch()
        test_accuracy = sess.run(accuracy,feed_dict={data_tensor:blob_test['data'].reshape(-1, 256*256), label_tensor:blob_test['class_labels']})
        print("step %d, test accuracy %g"%(i, test_accuracy))

# final validation
blob = val_io.fetch_batch()
print("Final test accuracy %g"%accuracy.eval(feed_dict={data_tensor: blob['data'].reshape(-1, 256*256), label_tensor: blob['class_labels']}))

print('Run `tensorboard --logdir=%s` in terminal to see the results.' % 'log')
