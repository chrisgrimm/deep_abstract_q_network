import numpy as np
import batch_helpers as bh
import vis_helpers as vh
import network

def train_vae(num_steps=-1, batch_size=32, disp_interval=100, save_interval=10000, network_name='vae_net'):
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    while i < num_steps:
        images = np.reshape(bh.get_batch(batch_size), [-1, 84, 84, 1])
        [_, loss, recon] = network.sess.run([network.train_op, network.loss, network.mu_x],
                                            feed_dict={network.inp_image: images / 255.})
        print i, loss
        if i % disp_interval == 0:
            vh.display_side_by_side('recent', images[0], recon[0] * 255.)
        if i % save_interval == 0:
            network.saver.save(network.sess, './'+network_name+'.ckpt')
        i += 1

train_vae()