import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

zipfile = np.load('./results/MNIST_data_target_activations.npz')
data_arr = zipfile['data_arr']
target_arr = zipfile['target_arr']
h1_arr = zipfile['h1_arr']
h2_arr = zipfile['h2_arr']
zipfile2 = np.load('./results/TSNE_pos.npz')
tsne_loc = zipfile2['tsne_pos']

h2_arr_pos = h2_arr*(h2_arr>=0)
plt.figure()
plt.subplot(2,5,1)
# sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr[:,0],alpha=0.6)
# plt.title('Original activations of Final layer Neuron 0')
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,0],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 0')
plt.subplot(2,5,2)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,1],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 1')
plt.subplot(2,5,3)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,2],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 2')
plt.subplot(2,5,4)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,3],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 3')
plt.subplot(2,5,5)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,4],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 4')
plt.subplot(2,5,6)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,5],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 5')
plt.subplot(2,5,7)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,6],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 6')
plt.subplot(2,5,8)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,7],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 7')
plt.subplot(2,5,9)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,8],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 8')
plt.subplot(2,5,10)
sns.scatterplot(x=tsne_loc[:,0],y=tsne_loc[:,1],hue=h2_arr_pos[:,9],alpha=0.6)
plt.title('Filtered activations of Final layer Neuron 9')
plt.show()