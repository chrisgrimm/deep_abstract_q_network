import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from numpy import linalg as LA
import time 



class FcnVAE(object): 

	def __init__(
		self, 
		dataset,
		n_z=20, 
		input_size=20*200, 
		batch_size=100, 
		learning_rate=.000005, 
		logs_path="/home/evan/Desktop/research/variational-autoencoder/models/log/",
		ckpt_dir = "/home/evan/Desktop/research/variational-autoencoder/models/train"
		): 

		self.n_z = n_z
		self.input_size = input_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dataset = dataset
		self.logs_path = logs_path
		self.ckpt_dir = ckpt_dir
		self.build_vae() 

	def _weights(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def _bias(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def _create_placeholders(self):
		with tf.name_scope('placeholders'): 
			self.x = tf.placeholder("float", shape=[None, self.input_size])

	def _create_feed_forward(self): 
		with tf.name_scope('feedforward'): 
			self.W_fc1 = self._weights([self.input_size, 500])
			self.b_fc1 = self._bias([500])
			self.h_1   = tf.nn.relu(tf.matmul(self.x, self.W_fc1) + self.b_fc1)
			self.W_fc2 = self._weights([500, 501]) #501 and not 500 to spot errors
			self.b_fc2 = self._bias([501])
			self.h_2   = tf.nn.relu(tf.matmul(self.h_1, self.W_fc2) + self.b_fc2)

	def _compute_gaussian_params(self): 
		with tf.name_scope('gaussian'): 
			self.z_mean = tf.add(tf.matmul(self.h_2, self._weights([501, self.n_z])), self._bias([self.n_z]))
			self.z_log_sigma_sq = tf.add(tf.matmul(self.h_2, 
				self._weights([501, self.n_z])), self._bias([self.n_z]))
			self.eps = tf.random_normal(
			(self.batch_size, self.n_z), 0, 1, dtype=tf.float32) # Adding a random number
			self.z = tf.add(self.z_mean, tf.multiply(
			tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps))  # The sampled z

	def _generate_from_gaussian(self): 
		with tf.name_scope('generator'): 
			self.W_fc1_g = self._weights([self.n_z, 500])
			self.b_fc1_g = self._bias([500])
			self.h_1_g   = tf.nn.relu(tf.matmul(self.z, self.W_fc1_g) \
			 + self.b_fc1_g)
			self.W_fc2_g = self._weights([500, 501])
			self.b_fc2_g = self._bias([501])
			self.h_2_g   = tf.nn.relu(tf.matmul(self.h_1_g, self.W_fc2_g) \
			 + self.b_fc2_g)

	def _compute_loss(self): 
		with tf.name_scope('loss'): 
			self.x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(self.h_2_g, 
				self._weights([501, self.input_size])), self._bias([self.input_size])))
			self.reconstr_loss = -tf.reduce_sum(self.x * tf.log(
				self.x_reconstr_mean + 1e-7) + (1-self.x) * tf.log( 
				1 - self.x_reconstr_mean + 1e-8), 1)
			self.latent_loss = -0.5 * tf.reduce_sum(
				1 + self.z_log_sigma_sq - tf.square(
					self.z_mean) - tf.exp(
					self.z_log_sigma_sq), 1)
			self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)   # average over batch

	def _opt_and_summaries(self): 
		with tf.name_scope('summaries'): 
			tf.summary.scalar("cost", self.cost)
			self.summary_op = tf.summary.merge_all() 
		with tf.name_scope('optimizer'): 
			self.optimizer = tf.train.AdamOptimizer(
				learning_rate=self.learning_rate).minimize(self.cost)

	def build_vae(self): 
		self._create_placeholders()
		self._create_feed_forward()
		self._compute_gaussian_params()
		self._generate_from_gaussian()
		self._compute_loss() 
		self._opt_and_summaries()

	def train(self, epochs): 
		init = tf.initialize_all_variables() 
		with tf.Session() as sess: 
			sess.run(init) 
			saver = tf.train.Saver()
			n_samples = self.dataset.num_examples
			batch_xs = self.dataset.next_batch(self.batch_size)
			start_cost = sess.run([self.cost], feed_dict={self.x: batch_xs})
			print('Test run after starting{}'.format(start_cost))
			writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
			for epoch in range(epochs): 
				avg_cost = 0.
				total_batch = int(n_samples / self.batch_size)
				# Loop over all batches
				for i in range(total_batch):
					batch_xs = self.dataset.next_batch(self.batch_size)
					_, batch_cost, summary = sess.run((
						self.optimizer, 
						self.cost, 
						self.summary_op), feed_dict={self.x: batch_xs})
					avg_cost += batch_cost / n_samples * self.batch_size

					writer.add_summary(summary, epoch * self.batch_size + i)
				# Display logs per epoch step
				if epoch % 10 == 0:
					save_path = saver.save(sess, self.ckpt_dir + "%d" % epoch + 'samples_x_coord') #Saves the weights (not the graph)
					print("Model saved in file: {}".format(save_path))
					#print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
				if epoch % 1 == 0:
					# save_path = saver.save(sess, ckpt_dir + "%d" % epoch + '20z_infepoch') #Saves the weights (not the graph)
					# print("Model saved in file: {}".format(save_path))
					print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

	def test_and_visulaize(ckpt_file, ckpt_dir): 
		saver = tf.train.Saver()
		with tf.Session() as sess:

			saver.restore(sess, ckpt_f)
			print("Model restored.")
			################################################################
			# Check norm problem on data with corresponding x coords 

			true_images = np.load('random_samples_with_x_coord.npy') 
			true_x_coord = np.load('random_samples_coord.npy')
			visualization = data.next_batch(batch_size)
			reshaped_vis = visualization.reshape(batch_size,20,200)
			
			mean = []
			mean_norm = []
			log_sigma_sq = []
			norm_log_sigma_sq = []
			xdist = []
			actual_means_np = np.zeros((10000, 20))
			for i in range(0, 8000, 20):
				x_sample = true_images[i:i + 20]
				#import pdb; pdb.set_trace()
				x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val  = sess.run(
					(x_reconstr_mean,z, z_mean, z_log_sigma_sq), feed_dict={x: x_sample})

				for j in range(20): 
					#actual_means_np[j + i] = z_mean_val[j]
					mean.append(np.average(z_mean_val[j]))
					mean_norm.append(LA.norm(z_mean_val[j]))
					log_sigma_sq.append(np.average(z_log_sigma_sq_val[j]))
					norm_log_sigma_sq.append(LA.norm(z_log_sigma_sq_val[j])) 
					xdist.append(true_x_coord[j + i][0])

			# plt.hist(mean_norm)
			# plt.title('Histogram of norm of mean_val, sample=8000')
			# plt.show() 
			import pdb; pdb.set_trace()
			plt.hist(mean)
			plt.title('Histogram of mean_val, sample=8000')
			plt.show() 
			plt.hist(log_sigma_sq)
			plt.title('Histogram of log_sigma_sq, sample=8000')
			plt.show() 
			plt.hist(norm_log_sigma_sq)
			plt.title('Histogram of norm of log_sigma_sq, sample=8000')
			plt.show() 

			plt.scatter(xdist, mean_norm)
			plt.title('Scatter of distance vs mean_norm')
			plt.show()
			plt.scatter(xdist, mean)
			plt.title('Scatter of distance vs mean')
			plt.show()
			plt.scatter(xdist, log_sigma_sq)
			plt.title('Scatter of distance vs log_sigma_sq')
			plt.show()
			plt.scatter(xdist, norm_log_sigma_sq)
			plt.title('Scatter of distance vs norm_log_sigma_sq')
			plt.show()


			
			

			#########################################################################################
			# check the norm and distances for the test linear sample 
			linear_vis = np.load('linear_samples_flatten.npy')
			x_dist = [168, 0, 40, 53, 120, 90, 0, 0, 118, 125, 20, 145, 5, 100]
			x_sample = data.next_batch(20)
			x_reconstruct,z_vals,z_mean_val,z_log_sigma_sq_val  = sess.run(
				(x_reconstr_mean,z, z_mean, z_log_sigma_sq), feed_dict={x: x_sample})

			x_linear_sample = data.next_batch(20)
			for i in range(linear_vis.shape[0]): 
				x_linear_sample[i] = linear_vis[i]

			xs_reconstruct,zs_vals,zs_mean_val,zs_log_sigma_sq_val  = sess.run(
				(x_reconstr_mean,z, z_mean, z_log_sigma_sq), feed_dict={x: x_linear_sample})

			fig = plt.figure() 
			x_norm_mean = []
			for i in range(len(x_dist)): 
				x_norm_mean.append(np.average(z_mean_val[i]))
			plt.scatter(x_dist, x_norm_mean)
			plt.show()
			##########################################################################################
			# Latent space for normal data
			plt.figure(figsize=(8, 12))
			for i in range(5):
				plt.subplot(5, 3, 3*i + 1)
				plt.imshow(x_sample[i].reshape(20, 200), 
					vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
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
				plt.imshow(x_reconstruct[i].reshape(20, 200), 
					vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
				plt.title("Reconstruction")
				#plt.colorbar()
				
			plt.tight_layout()
			plt.show()
			##########################################################################################
			#Latent Space for zs_means 
			########################################################################################
			plt.figure(figsize=(8, 12))
			for i in range(5):
				plt.subplot(5, 3, 3*i + 1)
				plt.imshow(x_linear_sample[i].reshape(20, 200), 
					vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
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
				plt.imshow(xs_reconstruct[i].reshape(20, 200),
				 vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
				plt.title("Reconstruction")
				#plt.colorbar()
			plt.tight_layout()
			plt.show()


class DataSet(object):
	def __init__(self):

		self._images = np.load('random_samples_with_x_coord.npy')
		self._images_x_coord = np.load('random_samples_coord.npy')
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



def main(): 
	data = DataSet() 
	fcn_vae = FcnVAE(data)
	fcn_vae.train(20)

if __name__ == '__main__':
	main()