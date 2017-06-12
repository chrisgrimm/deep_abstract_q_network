import tensorflow as tf 
import numpy as np 

def down_convolution_weights(inp, dqn_numbers, max_dqn_number, kernel, stride, filter_in, filter_out, rectifier):
    batch_size = tf.shape(inp)[0]
    inp = tf.reshape(inp, [batch_size] + [x.value for x in inp.get_shape()[1:]])
    #Ws , Bs = [], []
    #for i in range(max_dqn_number):
    #    Ws.append(tf.get_variable('w'+str(i), [kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer()))
    #    Bs.append(tf.get_variable('b'+str(i), [filter_out], initializer=tf.constant_initializer(0.0)))
    with tf.variable_scope('conv_vars'):
        W = tf.get_variable('w', [max_dqn_number, kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable('b', [max_dqn_number, filter_out], initializer=tf.constant_initializer(0.0))
    w = tf.reshape(tf.gather_nd(W, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, kernel, kernel, filter_in, filter_out])
    b = tf.reshape(tf.gather_nd(B, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, filter_out])
    print tf.expand_dims(inp, 0)
    c = tf.nn.conv3d(tf.expand_dims(inp, 0), w, [1, 1, stride, stride, 1], 'SAME')[0]# + tf.reshape(b, [batch_size, 1, 1, filter_out])
    print c
    #c = tf.reshape(c, [batch_size, tf.shape(c)[2], tf.shape(c)[3], tf.shape(c)[4]])
    print c
    return c, W, w, B

def down_convolution_weights2(inp, dqn_numbers, max_dqn_number, kernel, stride, filter_in, filter_out, rectifier):
    batch_size = tf.shape(inp)[0]
    inp = tf.reshape(inp, [batch_size] + [x.value for x in inp.get_shape()[1:]])
    #Ws , Bs = [], []
    #for i in range(max_dqn_number):
    #    Ws.append(tf.get_variable('w'+str(i), [kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer()))
    #    Bs.append(tf.get_variable('b'+str(i), [filter_out], initializer=tf.constant_initializer(0.0)))
    with tf.variable_scope('conv_vars'):
        W = tf.get_variable('w', [max_dqn_number, kernel, kernel, filter_in, filter_out], initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable('b', [max_dqn_number, filter_out], initializer=tf.constant_initializer(0.0))
    w = tf.reshape(tf.gather_nd(W, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, kernel, kernel, filter_in, filter_out])
    b = tf.reshape(tf.gather_nd(B, tf.reshape(dqn_numbers, [-1, 1])), [batch_size, filter_out])
    all_res = []
    for i in range(filter_out):
        wi = tf.transpose(w[:, :, :, :, i], [1, 2, 3, 0])
        bi = tf.transpose(b[:, i])
        input_i = tf.transpose(tf.expand_dims(inp[:, :, :, i], 0), [0, 2, 3, 1]) # add dummy batch_dim, focus on one channel at a time.
        print 'wi', wi
        print 'inp_i', input_i

        res = tf.nn.conv2d(input_i, wi, [1, stride, stride, 1], 'VALID') # [1, h/stride, w/stride, batch_size]
        all_res.append(res)
    c = tf.transpose(tf.concat(all_res, 0), [3, 1, 2, 0])
    #print tf.expand_dims(inp, 0)
    #c = tf.nn.conv3d(tf.expand_dims(inp, 0), w, [1, 1, stride, stride, 1], 'SAME')[0]# + tf.reshape(b, [batch_size, 1, 1, filter_out])
    #print c
    #c = tf.reshape(c, [batch_size, tf.shape(c)[2], tf.shape(c)[3], tf.shape(c)[4]])
    #print c
    return c, W, w, B




sess = tf.Session()
fake_inp = tf.random_uniform([2, 2, 2, 1], 0, 1)
dqn_numbers = [2, 4]
stride = 1
filter_size = 1
filter_in = 1
filter_out = 1
max_dqn_number = 5
out, W, w_slice, B = down_convolution_weights2(fake_inp, dqn_numbers, max_dqn_number, filter_size, stride, filter_in, filter_out, tf.nn.relu)
out_alt_1 = tf.nn.conv2d(tf.expand_dims(fake_inp[0], 0), W[2, :, :, :, :], [1, stride, stride, 1], 'SAME')# + B[2, :]
out_alt_2 = tf.nn.conv2d(tf.expand_dims(fake_inp[1], 0), W[4, :, :, :, :], [1, stride, stride, 1], 'SAME')# + B[4, :]
out_alt = tf.concat([out_alt_1, out_alt_2], 0)
w_slice_alt_1 = W[2, :, :, :, :]
w_slice_alt_2 = W[4, :, :, :, :]
w_slice_alt = tf.concat([tf.expand_dims(x, 0) for x in [w_slice_alt_1, w_slice_alt_2]], 0)

sess.run(tf.initialize_all_variables())

[real_out, real_out_alt] = sess.run([out, out_alt])
[w_slice_out, w_slice_out_alt] = sess.run([w_slice, w_slice_alt])

print w_slice_out.shape
print w_slice_out_alt.shape
print (w_slice_out == w_slice_out_alt).all()
print ((real_out - real_out_alt)**2 < 10**-5)
print real_out.shape
print real_out_alt.shape
