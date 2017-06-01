import matplotlib.pyplot as plt
import numpy as np
import batch_helpers as bh
import toy_mr

def display_side_by_side(save_name, image, recon):
    f, axs = plt.subplots(1, 2)
    axs[0].imshow(np.tile(image, [1, 1, 3]))
    axs[1].imshow(np.tile(recon, [1, 1, 3]))
    f.savefig('./'+save_name+'.png')

def encoding_to_state(state, env, name):
    walls = []
    doors = []
    traps = []
    keys = []
    agents = []
    for x in range(11):
        for y in range(11):
            if round(state[x, y, toy_mr.WALL_CODE-1]) == 1:
                walls.append((x, y))
            if round(state[x, y, toy_mr.DOOR_CODE-1]) == 1:
                doors.append((x, y))
            if round(state[x, y, toy_mr.TRAP_CODE-1]) == 1:
                traps.append((x, y))
            if round(state[x, y, toy_mr.KEY_CODE-1]) == 1:
                keys.append((x, y))
            if round(state[x, y, toy_mr.AGENT_CODE-1]) == 1:
                agents.append((x, y))

    for x in range(11):
        if round(state[x, y, toy_mr.KEY_CODE - 1]) == 1:
            keys.append((x, y))

    env.render_screen_generated(name, walls, keys, doors, traps, agents)

def visualize_toy_mr(name='prob_visualization'):
    import network_fc as network
    # get a set of all the agent positions in the room
    state = bh.get_toy_mr_batch(32)[0]
    # pass them to the network and normalize their rarities
    # replace the white squares in the room with rarities
    positions = []
    # remove the agent.
    state[:, :, toy_mr.AGENT_CODE-1] = 0
    for x in range(11):
        for y in range(11):
            if np.sum(state[x, y]) == 0:
                positions.append((x,y))
    new_states = []
    for (x,y) in positions:
        new_state = np.copy(state)
        new_state[x, y, :] = 0
        new_state[x, y, toy_mr.AGENT_CODE-1] = 1
        new_states.append(new_state)

    #network.saver.restore(network.sess, './'+network_name+'.ckpt')
    [p_x, mse, var] = network.sess.run([network.p_x, network.mse, network.var], feed_dict={network.inp_image: new_states})
    normed_mse = (mse - np.min(mse))/(np.max(mse) - np.min(mse))
    normed_var = (var - np.min(var))/(np.max(var) - np.min(var))
    rarity = normed_mse * normed_var + (1 - normed_mse) * np.max(normed_var)
    normed_rarity = (rarity - np.min(rarity))/(np.max(rarity) - np.min(rarity))
    precision = 50
    max_prob = np.max(p_x)
    normalized_prob_x = np.exp(p_x - max_prob)
    #temp = np.abs(np.min(prob_x))
    #normalized_prob_x = np.exp(prob_x/temp)
    #normalized_prob_x = prob_x
    #normalized_prob_x = (normalized_prob_x - np.min(normalized_prob_x)) / (np.max(normalized_prob_x) - np.min(normalized_prob_x))
    #normalized_rarities = []
    #for r in rarity:
    #    normalized_rarities.append(np.exp(temp/x))
    #normalized_rarities = np.array(normalized_rarities)
    #normalized_rarities = normalized_rarities
    #normalized = (z_variances - np.min(z_variances)) / np.max(z_variances)
    #normed_losses = (losses - np.min(losses)) / np.max(losses)
    #intensities = np.clip(normed_losses * -1 + (1 - normed_losses)*normalized, 0, 1)

    intensities = [tuple([int(255*(1-n))]*3) for n in normed_rarity]
    bh.toy_mr_env.draw_probability_screen(name, positions, intensities)

