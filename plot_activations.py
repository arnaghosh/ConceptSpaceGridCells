import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.random.seed(13)
zipfile = np.load('./results/MNIST_data_target_activations.npz')
data_arr = zipfile['data_arr']
target_arr = zipfile['target_arr']
h1_arr = zipfile['h1_arr']
h2_arr = zipfile['h2_arr']
# zipfile2 = np.load('./results/TSNE_pos.npz')
zipfile2 = np.load('./results/MNIST_network_space_tsne_pos.npz')
tsne_loc = zipfile2['tsne_pos']

h1_arr_pos = h1_arr
h2_arr_pos = h2_arr*(h2_arr>=0)
start_neuron=25
vmin=h1_arr_pos.min()
vmax=h1_arr_pos.max()
plt.figure()
for i in range(25):
	plt.subplot(5,5,i+1)
	sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h1_arr_pos[:,start_neuron+i], alpha=0.6)
	# plt.scatter(x=tsne_loc[:,0],y=tsne_loc[:,1],c=h1_arr_pos[:,start_neuron+i], alpha=0.6,vmin=vmin,vmax=vmax)
	plt.title('Filtered activations of Pre-Final layer Neuron '+str(start_neuron+i))
# plt.legend()
plt.show()