import tensorflow as tf
import numpy as np
#from transformer import transformer
from tensorflow.contrib.layers import batch_norm

def binarizer(x, num_bits, batch_size):
    x_size = x.get_shape()[1].value
    w_bin = tf.get_variable('wbin', shape=[x_size, num_bits],
                           initializer=tf.contrib.layers.xavier_initializer())
    b_bin = tf.get_variable('bbin', shape=[num_bits], initializer=tf.constant_initializer(0.0))
    b = tf.nn.tanh(tf.matmul(x, w_bin) + b_bin)
    noise = tf.random_uniform([batch_size, num_bits])
    eps = tf.stop_gradient(tf.select(tf.less_equal(noise, (1 + b)/2), 1 - b, -b - 1))
    return b + eps


def leakyRelu(x, alpha=0.001):
    return tf.maximum(alpha * x, x)


def fullyConnected(inp_layer, numNeurons, rectifier=leakyRelu, bias=1.0):
    assert len(inp_layer.get_shape()) == 2
    inp_dim = inp_layer.get_shape()[1].value
    w1 = tf.get_variable('w1', shape=[inp_dim, numNeurons],
                        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[numNeurons],
                        initializer=tf.constant_initializer(bias))
    if rectifier:
        h1 = rectifier(tf.matmul(inp_layer, w1) + b1)
    else:
        h1 = tf.matmul(inp_layer, w1) + b1
    return h1

def embeddingLookup(inp_layer, context_size, embed_size, vocab_size, normalize=False):
    E = tf.get_variable('E', shape=[vocab_size, embed_size], initializer=tf.contrib.layers.xavier_initializer())
    if not normalize:
        e1 = tf.matmul(inp_layer, E) * (1.0 / context_size)
    else:
        embedding_sum = tf.matmul(inp_layer, E)
        norm = tf.reduce_sum(inp_layer, reduction_indices=[1])
        norm = tf.tile(tf.expand_dims(norm, 1), [1, embed_size])
        e1 = tf.div(embedding_sum, norm)
    return e1

def makeMask(h, w, scale=2):
    mask = np.zeros((scale*h, scale*w), dtype=np.float32)
    indices = [(wi, hi) for wi in range(scale*w) for hi in range(scale*h)]
    for wi, hi in indices:
        mask[wi, hi] = 1.0
    return mask

def unpoolLayer(inp_layer, scale=2):
    assert len(inp_layer.get_shape()) == 4
    [h, w, d] = [x.value for x in inp_layer.get_shape()[1:]]
    mask = tf.tile(tf.reshape(makeMask(h, w, scale=scale), [scale*h, scale*w, 1]), [1, 1, d])
    return mask * tf.image.resize_nearest_neighbor(inp_layer, (scale*h, scale*w))

def upConvolution(inp_layer, filter_size, filter_in, filter_out, rectifier=leakyRelu, bias=0.3, scale=2, use_batch_norm=False, is_training=True):
    assert len(inp_layer.get_shape()) == 4
    [h, w, d] = [x.value for x in inp_layer.get_shape()[1:]]
    #unpooled = unpoolLayer(inp_layer, scale)
    #w1 = tf.get_variable('w1', shape=[filter_size, filter_size, filter_in, filter_out],
    #                     initializer=tf.contrib.layers.xavier_initializer())
    #b1 = tf.get_variable('b1', shape=[filter_out], initializer=tf.constant_initializer(bias))
    #conv = tf.nn.conv2d(unpooled, w1, [1, 1, 1, 1], 'SAME') +b1
    w1 = tf.get_variable('w1', shape=[filter_size, filter_size, filter_out, filter_in],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[filter_out], initializer=tf.constant_initializer(bias))
    conv = tf.nn.conv2d_transpose(inp_layer, w1, [32, 2*h, 2*w, filter_out], [1, 2, 2, 1]) + b1
    #print '!!!'
    #print inp_layer
    #conv = tf.nn.atrous_conv2d(inp_layer, w1, 3, 'SAME') + b1
    #print conv
    if use_batch_norm:
        normed = batch_norm(conv, is_training=is_training)
    else:
        normed = conv
    c1 = rectifier(normed)
    return c1

def downConvolution(inp_layer, filter_size, pool_size, filter_in, filter_out, conv_stride=1, pool_stride=None, rectifier=leakyRelu, pool=True, use_batch_norm=False, is_training=True):
    if not pool_stride:
        pool_stride = pool_size

    assert len(inp_layer.get_shape()) == 4
    w1 = tf.get_variable('w1', shape=[filter_size, filter_size, filter_in, filter_out],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[filter_out], initializer=tf.constant_initializer(0.1))
    #c1 = tf.nn.relu(tf.nn.conv2d(inp_layer, w1, [1, 1, 1, 1], 'SAME') + b1)
    conv = tf.nn.conv2d(inp_layer, w1, [1, conv_stride, conv_stride, 1], 'SAME') + b1
    if use_batch_norm:
        normed = batch_norm(conv, is_training=is_training)
    else:
        normed = conv
    #normed = conv
    p1 = rectifier(normed)
    #if pool:
    #    p1 = tf.nn.max_pool(p1, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1], 'SAME')
    return p1

def produce_affine_transform(scale, trans):
    fc1 = tf.concat(1, [scale, trans])
    affine_transform = tf.transpose(tf.gather(tf.transpose(fc1), np.mat('0 0 2; 1 1 3')), perm=[2, 0, 1]) * np.mat('1 0 1; 0 1 1')
    return affine_transform

def objectVolume(inp_image, batch_size):
    # 64 x 64 x 3
    with tf.variable_scope('c1'):
        c1 = downConvolution(inp_image, 5, 2, 3, 6) # 32 x 32 x 6
    with tf.variable_scope('c2'):
        c2 = downConvolution(c1, 5, 2, 6, 12) # 16 x 16 x 12
    with tf.variable_scope('c3'):
        c3 = downConvolution(c2, 5, 2, 12, 24) # 8 x 8 x 24
    with tf.variable_scope('c4'):
        c4 = downConvolution(c3, 3, 2, 24, 36) # 4 x 4 x 36
    with tf.variable_scope('scale'):
        scale = fullyConnected(tf.reshape(c4, [32, 4*4*36]), 2, rectifier=lambda x: x, bias=1.0)
    with tf.variable_scope('trans'):
        trans = fullyConnected(tf.reshape(c4, [32, 4*4*36]), 2, rectifier=lambda x: x, bias=0.0)
    affine = produce_affine_transform(scale, trans)
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    y = np.linspace(-1, 1, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    meshgrid = tf.tile(tf.expand_dims(tf.concat(2, [tf.expand_dims(tf.to_float(xx), -1), tf.expand_dims(tf.to_float(yy), -1)]), 0), [batch_size, 1, 1, 1])
    augmented_image = tf.concat(3, [inp_image, meshgrid])
    print augmented_image
    window = transformer(augmented_image, tf.reshape(affine, [32, 6]), (64, 64))
    top_left = window[:, 0, 0, 3:]
    bottom_right = window[:, 63, 63, 3:]
    print top_left, bottom_right
    return tf.reshape(top_left, [32, 2]), tf.reshape(bottom_right, [32, 2])

def grid_layer(frame, batch_size, as_image=False):
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    y = np.linspace(-1, 1, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    #meshgrid = tf.tile(tf.expand_dims(tf.concat(2, [tf.expand_dims(tf.to_float(xx), -1), tf.expand_dims(tf.to_float(yy), -1)]), 0), [batch_size, 1, 1, 1])
    #augmented = tf.concat(3, [frame, meshgrid])
    with tf.variable_scope('c1'):
        w = tf.get_variable('w', [7, 7, 3, 3], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [3], initializer=tf.constant_initializer(0.5))
        h = leakyRelu(tf.nn.conv2d(frame, w, [1, 1, 1, 1], 'SAME') + b)
    with tf.variable_scope('c2'):
        w = tf.get_variable('w', [7, 7, 3, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.5))
        h = leakyRelu(tf.nn.conv2d(h, w, [1, 1, 1, 1], 'SAME') + b)
    #with tf.variable_scope('c3'):
    #    w = tf.get_variable('w', [7, 7, 20, 1], initializer=tf.contrib.layers.xavier_initializer())
    #    b = tf.get_variable('b', [1], initializer=tf.random_uniform_initializer(0.1, 0.5))
    #    h = leakyRelu(tf.nn.conv2d(h, w, [1, 1, 1, 1], 'SAME') + b)

    #with tf.variable_scope('fc1'):
    #    fc1 = fullyConnected(tf.reshape(frame, [-1, 64*64*3]), 100)
    #with tf.variable_scope('fc2'):
    #    h = tf.reshape(fullyConnected(fc1, 64*64), [-1, 64, 64, 1])
    probs = tf.reshape(tf.nn.softmax(tf.reshape(h, [-1, 64*64])), [-1, 64, 64])
    print probs.get_shape()
    x = tf.reduce_sum(probs * xx, reduction_indices=[1, 2])
    y = tf.reduce_sum(probs * yy, reduction_indices=[1, 2])
    var_x = tf.reduce_sum(probs * np.power(xx, 2), reduction_indices=[1,2]) - tf.pow(x, 2)
    var_y = tf.reduce_sum(probs * np.power(yy, 2), reduction_indices=[1,2]) - tf.pow(y, 2)
    variance = var_x + var_y
    xy = tf.concat(1, [tf.expand_dims(x, -1), tf.expand_dims(y, -1)])
    if as_image:
        return xy, probs, variance
    else:
        return xy, variance


    #if not as_image:
    #    return tf.concat(1, [tf.expand_dims(x, -1), tf.expand_dims(y, -1)]), probs, variance
    #else:
    #    return kernel_image(x, y, 64), tf.concat(1, [tf.expand_dims(x, -1), tf.expand_dims(y, -1)]), probs

def kernel_image(coord_x, coord_y, dim):
    x = np.linspace(-1, 1, dim, dtype=np.float32)
    y = np.linspace(-1, 1, dim, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    coord_x = tf.tile(tf.reshape(coord_x, [-1, 1, 1]), [1, dim, dim])
    coord_y = tf.tile(tf.reshape(coord_y, [-1, 1, 1]), [1, dim, dim])
    return tf.stop_gradient(tf.exp(-(tf.pow(coord_x - xx, 2) + tf.pow(coord_y - yy, 2))))

def kernel_channels(coords, dim):
    channels = []
    for coord in coords:
        channels.append(tf.expand_dims(kernel_image(coord[:, 0], coord[:, 1], dim), -1))
    return tf.concat(3, channels)

def importance_map(model_in, model_hook, model_scope, image_size, noise=None, mask_scope='mask'):
    with tf.variable_scope(mask_scope):
        with tf.variable_scope('fc1'):
            fc1 = fullyConnected(tf.reshape(model_in, [-1, image_size*image_size*3]), 100)
        with tf.variable_scope('fc3'):
            mask_var = tf.reshape(fullyConnected(fc1, image_size*image_size, rectifier=tf.nn.sigmoid, bias=-1.0), [-1, image_size, image_size, 1])
        mask = tf.tile(mask_var, [1, 1, 1, 3])
        # white means use noise, black means use signal
        if noise == None:
            noise_image = tf.random_uniform(model_in.get_shape(), 0, 1)
        else:
            noise_image = noise
        inp_image_noise =  noise_image * mask + model_in * (1 - mask)

    with tf.variable_scope(model_scope, reuse=True):
        noise_out = model_hook(inp_image_noise)

    return mask, noise_out


def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
