
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from numpy import linalg as LA

logs_path = "/home/evan/Desktop/research/variational-autoencoder/models/log/"

def show(arr): 
	toimage(arr).show()

ckpt_dir = "/home/evan/Desktop/research/variational-autoencoder/models/train"

class DataSet(object):
	def __init__(self):

		self._images = np.load('200x20_processed.npy')
		self._num_examples = self._images.shape[0]
		self._epochs_completed = 0
		self._index_in_epoch = 0
		perm = np.arange(self._num_examples)
		np.random.shuffle(perm)
		self._images = self._images[perm]


	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end]




def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

n_z = 20 #Dimension of the latent space
# Input
x = tf.placeholder("float", shape=[None, 20 * 200]) #Batchsize x Number of Pixels
#y_ = tf.placeholder("float", shape=[None, 10])   #Batchsize x 10 (one hot encoded)

# First hidden layer
W_fc1 = weights([20 * 200, 500])
b_fc1 = bias([500])
h_1   = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

# Second hidden layer 
W_fc2 = weights([500, 501]) #501 and not 500 to spot errors
b_fc2 = bias([501])
h_2   = tf.nn.relu(tf.matmul(h_1, W_fc2) + b_fc2)

# Parameters for the Gaussian
z_mean = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))
z_log_sigma_sq = tf.add(tf.matmul(h_2, weights([501, n_z])), bias([n_z]))

batch_size = 20
eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32) # Adding a random number
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))  # The sampled z

W_fc1_g = weights([n_z, 500])
b_fc1_g = bias([500])
h_1_g   = tf.nn.relu(tf.matmul(z, W_fc1_g) + b_fc1_g)

W_fc2_g = weights([500, 501])
b_fc2_g = bias([501])
h_2_g   = tf.nn.relu(tf.matmul(h_1_g, W_fc2_g) + b_fc2_g)
x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(h_2_g,  weights([501, 20 * 200])), bias([20 * 200])))

# reconstr_loss = -tf.reduce_sum(x * tf.log(
# 	tf.clip_by_value(x_reconstr_mean, 1e-8, 1.0) ) + (1-x) * tf.log( 
# 	tf.clip_by_value(1 - x_reconstr_mean, 1e-8, 1.0)), 1)

reconstr_loss = -tf.reduce_sum(x * tf.log(
	x_reconstr_mean + 1e-7) + (1-x) * tf.log( 
	1 - x_reconstr_mean + 1e-8), 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch

#tf.summary.scalar("reconstr_loss", reconstr_loss)
#tf.summary.scalar("latent_loss", latent_loss)
tf.summary.scalar("cost", cost)
# Use ADAM optimizer
optimizer =  tf.train.AdamOptimizer(learning_rate=0.000005).minimize(cost)

summary_op = tf.summary.merge_all()

runs = 500 #Set to 0, for no training
init = tf.initialize_all_variables()
saver = tf.train.Saver()
data = DataSet()
n_samples = data.num_examples



with tf.Session() as sess:
	sess.run(init)
	batch_xs = data.next_batch(batch_size)
	print(batch_xs.shape)
	dd = sess.run([cost], feed_dict={x: batch_xs})
	print('Test run after starting {}'.format(dd))
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) 

	for epoch in range(runs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = data.next_batch(batch_size)
			_,d, summary = sess.run((optimizer, cost, summary_op), feed_dict={x: batch_xs})
			avg_cost += d / n_samples * batch_size

			writer.add_summary(summary, epoch * batch_size + i)
			#print avg_cost

		# Display logs per epoch step
		if epoch % 10 == 0:
			save_path = saver.save(sess, ckpt_dir + "%d" % epoch + 'inf_epoch_lr_large') #Saves the weights (not the graph)
			print("Model saved in file: {}".format(save_path))
			#print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
		if epoch % 1 == 0:
			# save_path = saver.save(sess, ckpt_dir + "%d" % epoch + '20z_infepoch') #Saves the weights (not the graph)
			# print("Model saved in file: {}".format(save_path))
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

# import pdb; pdb.set_trace()
#check_point_file = "model/model.ckpt"
#check_point_file = "/Users/oli/Dropbox/Server_Sync/vae/models/model_200.ckpt"
ckpt_f = '/home/evan/Desktop/research/variational-autoencoder/models/train40inf_epoch_lr_large'

#import pdb; pdb.set_trace()
saver = tf.train.Saver()
with tf.Session() as sess:
	#import pdb; pdb.set_trace() 
	visualization = data.next_batch(batch_size)
	reshaped_vis = visualization.reshape(batch_size,20,200)
	
	####################################################
	#ims("results/base.jpg",merge(reshaped_vis[:8],[20, 200]))
	# train

	# to keep farthest 0
	# close 1 
	# medium 3
	# import pdb; pdb.set_trace()

	linear_vis = np.load('linear_samples_flatten.npy')
	# visualization[0] = linear_vis[0]  168
	# visualization[1] = linear_vis[1]   0
	# visualization[2] = linear_vis[3]  53 
	"""
	0: 168
	1: 0
	2: 40
	3: 53
	4: 120 
	5: 90 
	6: 0
	7: 0
	8: 180
	9: 125
	10: 20
	11: 145 
	12: 5
	13: 100 

	"""
	x_dist = [168, 0, 40, 53, 120, 90, 0, 0, 118, 125, 20, 145, 5, 100]
	# show(visualization[0])  169
	# show(visualization[1])  0 
	# show(visualization[2])  53 
	saver.restore(sess, ckpt_f)
	print("Model restored.")
	x_sample = data.next_batch(20)
	x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val  = sess.run((x_reconstr_mean,z, z_mean, z_log_sigma_sq), feed_dict={x: x_sample})

	x_linear_sample = data.next_batch(20)

	for i in range(linear_vis.shape[0]): 
		x_linear_sample[i] = linear_vis[i]

	xs_reconstruct,zs_vals,zs_mean_val,zs_log_sigma_sq_val  = sess.run((x_reconstr_mean,z, z_mean, z_log_sigma_sq), feed_dict={x: x_linear_sample})

	fig = plt.figure() 
	x_norm_mean = []
	for i in range(len(x_dist)): 
		x_norm_mean.append(LA.norm(zs_mean_val[i]))
	plt.scatter(x_dist, x_norm_mean)
	plt.show()
	# import pdb; pdb.set_trace()

	# toimage(x_sample[0].reshape(20, 200)).show()
	# toimage(x_reconstruct[0].reshape(20, 200)).show()
	plt.figure(figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 3, 3*i + 1)
		plt.imshow(x_sample[i].reshape(20, 200), vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
		plt.title("Test input")
		
		#plt.colorbar()
		plt.subplot(5, 3, 3*i + 2)
		plt.scatter(z_vals[:,0],z_vals[:,1], c='gray', alpha=0.5)
		plt.scatter(z_mean_val[i,0],z_mean_val[i,1], c='green', s=64, alpha=0.5)
		plt.scatter(z_vals[i,0],z_vals[i,1], c='blue', s=16, alpha=0.5)
	   
		plt.xlim((-8,8))
		plt.ylim((-8,8))
		plt.title("Latent Space")
		
		plt.subplot(5, 3, 3*i + 3)
		plt.imshow(x_reconstruct[i].reshape(20, 200), vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
		plt.title("Reconstruction")
		#plt.colorbar()
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 3, 3*i + 1)
		plt.imshow(x_linear_sample[i].reshape(20, 200), vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
		plt.title("Test input")
		
		#plt.colorbar()
		plt.subplot(5, 3, 3*i + 2)

		# plt.scatter(LA.norm(zs_mean_val[i]), 4, c='blue', alpha=.5)
		plt.scatter(zs_vals[:,0],zs_vals[:,1], c='gray', alpha=0.5)
		plt.scatter(zs_mean_val[i,0],zs_mean_val[i,1], c='green', s=64, alpha=0.5)
		plt.scatter(zs_vals[i,0],zs_vals[i,1], c='blue', s=16, alpha=0.5)
	   
		# plt.xlim((-8,8))
		# plt.ylim((-8,8))
		plt.title("Latent Space")
		
		plt.subplot(5, 3, 3*i + 3)
		plt.imshow(xs_reconstruct[i].reshape(20, 200), vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
		plt.title("Reconstruction")
		#plt.colorbar()
	plt.tight_layout()
	plt.show()

