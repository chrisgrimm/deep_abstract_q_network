import cv2, numpy as np, os, tqdm
from replay_memory import ReplayMemory
import itertools

num_actions = 3
def modify_image(image):
    #image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_LINEAR)
    #image = #np.rint(image[:, :, 0]*0.2989 + image[:, :, 1]*0.5870 + image[:, :, 2]*0.1140).astype(np.uint8)
    image = image[:, :, 0]
    return image


def extract_episode(episode_dir):
    with open(os.path.join(episode_dir, 'action.log')) as f:
        actions = np.array([int(line) for line in f.readlines()])
        actions_onehot = np.zeros((len(actions), num_actions), dtype=np.uint8)
        actions_onehot[range(len(actions)), actions] = 1
    with open(os.path.join(episode_dir, 'reward.log')) as f:
        rewards = np.array([int(line) for line in f.readlines()], dtype=np.uint8)
    frame_files = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
    frame_files = [f for f in frame_files if f.endswith('.png')]
    #for f in frame_files:
    #    cv2.imshow('f', cv2.resize(modify_image(cv2.imread(f)) / 255.0, (400, 400)))
    #    cv2.waitKey()
    frames = np.zeros((len(frame_files), 84, 84), dtype=np.uint8)
    for i, f in enumerate(frame_files):
        image = modify_image(cv2.imread(f))
        frames[i] = image
    return frames, actions_onehot, rewards

def extract_all_episodes(episode_dir):
    episodes = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
    episodes = [e for e in episodes if os.path.isdir(e)]
    all_frames, all_actions, all_rewards = [], [], []
    for episode in tqdm.tqdm(episodes):
        frames, actions, rewards = extract_episode(episode)
        all_frames.append(frames)
        all_actions.append(actions)
        all_rewards.append(rewards)
    return all_frames, all_actions, all_rewards

def extract_all_episodes_iter(episode_dir, cycle=False):
    episodes = [os.path.join(episode_dir, x) for x in os.listdir(episode_dir)]
    episodes = [e for e in episodes if os.path.isdir(e)]
    #all_frames, all_actions, all_rewards = [], [], []
    iter = itertools.cycle(episodes) if cycle else tqdm.tqdm(episodes)
    for episode in iter:
        frames, actions, rewards = extract_episode(episode)
        yield frames, actions, rewards
        #all_frames.append(frames)
        #all_actions.append(actions)
        #all_rewards.append(rewards)
    #return all_frames, all_actions, all_rewards

#all_frames, all_actions, all_rewards = extract_all_episodes(episode_dir)

def load_into_replay_memory(episode_dir):
    capacity = 500000
    replay_buffer = ReplayMemory([84, 84], 'uint8', capacity, 4)
    j = 0
    cap_counter = 0
    for frames, actions, rewards in extract_all_episodes_iter(episode_dir):
        ep_length = len(frames)-1
        for i in xrange(ep_length):
            s1 = frames[i]
            a = np.argmax(actions[i])
            r = rewards[i]
            s2 = frames[i+1]
            t = (i == (ep_length - 1))
            replay_buffer.append(s1, a, r, s2, t)
        cap_counter += ep_length
        print cap_counter
        if cap_counter > capacity:
            break
    return replay_buffer

print 'Loading replay memory...'
buffer = load_into_replay_memory('./train')
