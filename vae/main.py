import numpy as np
import batch_helpers as bh
import vis_helpers as vh
import wind_tunnel
import os
import tqdm
import matplotlib.pyplot as plt
import cv2



def train_vae(num_steps=-1, batch_size=32, disp_interval=100, save_interval=10000, network_name='vae_net'):
    import network as network
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    while i < num_steps:
        images = np.reshape(bh.get_batch(batch_size), [-1, 84, 84, 1])
        [_, loss, recon] = network.sess.run([network.train_op, network.loss, network.mu_x],
                                            feed_dict={network.inp_image: images / 255.})
        print i, loss
        if i % disp_interval == 0:
            vh.display_side_by_side('recent', images[0], recon[0])
        if i % save_interval == 0:
            network.saver.save(network.sess, './'+network_name+'.ckpt')
        i += 1
#@profile
def train_toy_mr_vae(num_steps=-1, disp_directory='./toy_mr_run_normal_enc100_rarity', batch_size=32, disp_interval=1000, save_interval=10000, network_name='vae_net_fc'):
    import network_fc as network
    if not os.path.isdir(disp_directory):
        os.mkdir(disp_directory)
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    #network.saver.restore(network.sess, './'+network_name+'.ckpt')
    while i < num_steps:
        states = bh.get_toy_mr_batch(batch_size)
        [_, loss, p_x, mu_x] = network.sess.run([network.train_op, network.loss, network.p_x, network.mu_x],
                                            feed_dict={network.inp_image: states})
        #states = bh.get_toy_mr_batch(batch_size)
        #[_, loss, rarity, p_x, sigma_x] = network.sess.run([network.train_op_dec, network.loss, network.rarity, network.p_x, network.sigma_x],
        #                                    feed_dict={network.inp_image: states})
        if np.any(np.isnan(p_x)):
            print 'AHHH!'

        print i, loss, p_x[0]
        if i % save_interval == 0:
            network.saver.save(network.sess, './' + network_name + '.ckpt')
        if i % disp_interval == 0:
            filename_gen = os.path.join(disp_directory, 'gen_image%s' % i)
            vh.encoding_to_state(mu_x[0], bh.toy_mr_env, filename_gen)
            filename_gt = os.path.join(disp_directory, 'gt_image%s' % i)
            vh.encoding_to_state(states[0], bh.toy_mr_env, filename_gt)
            filename = os.path.join(disp_directory, 'prob_vis_iter%s' % i)
            vh.visualize_toy_mr(name=filename)
        i += 1



def plot_variance_vs_position(result_dir='variance_vs_position', network_name='vae_net', var_max=5.0):
    import network as network
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    game = wind_tunnel.WindTunnel()
    encodings = []
    recons = []
    images = []
    losses = []
    network.saver.restore(network.sess, './'+network_name+'.ckpt')
    for i in tqdm.tqdm(range(100)):
        game.agent = i
        game.generate_new_state()
        image = np.reshape(np.array(game.get_current_state()), [-1, 84, 84, 1])
        images.append(image)
        [encoding_mean, recon, loss] = network.sess.run([network.mu_z, network.mu_x, network.loss], feed_dict={network.inp_image: image / 255.})
        encodings.append(encoding_mean)
        losses.append(loss)
        recons.append(recon)
    losses = np.array(losses)
    losses = losses - np.min(losses)
    losses = losses / np.max(losses)
    encodings = np.concatenate(encodings, 0)
    variances = np.sqrt(np.sum((encodings ** 2), axis=1))
    novelty = var_max * losses + variances * (1. - losses)
    f, ax = plt.subplots()
    ax.plot(range(100), variances, 'blue')
    ax.plot(range(100), losses, 'red')
    ax.plot(range(100), novelty, 'green')
    f.savefig(os.path.join(result_dir, 'variance_plot.png'))
    side_by_side_dir = os.path.join(result_dir, 'reconstructions')
    if not os.path.isdir(side_by_side_dir):
        os.mkdir(side_by_side_dir)
    for i in tqdm.tqdm(range(100)):
        vh.display_side_by_side(os.path.join(side_by_side_dir, '%5d' % i), images[i][0], recons[i][0])


train_toy_mr_vae()
#train_vae()
#plot_variance_vs_position(network_name='dqn_vae_net')