import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import Net

# load the dataset - only testset here
mnist_testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]))
data_loader = torch.utils.data.DataLoader(mnist_testset,shuffle=False,batch_size=1000)

# load the trained Network
network = Net()
network.load_state_dict(torch.load('./results/model.pth'))

# function for generating the activations of final and penultimate layer neurons
def generate_features(cuda=False):
	if cuda:
		network.cuda()
	network.eval()
	out_data = []
	out_target = []
	out_h1 = []
	out_h2 = []
	for data,target in tqdm(data_loader):
		data_np = data.numpy()
		target_np = target.numpy()
		if cuda:
			data = data.cuda()
			target = target.cuda()
		with torch.no_grad():
			output = network(data)
			out_h1_np = network.penultimate_layer_activations().cpu().numpy()
			out_h2_np = network.final_layer_activations().cpu().numpy()

		out_data.append(data_np)
		out_target.append(target_np)
		out_h1.append(out_h1_np)
		out_h2.append(out_h2_np)

	out_data_arr = np.concatenate(out_data,axis=0)
	out_target_arr = np.concatenate(out_target,axis=0)
	out_h1_arr = np.concatenate(out_h1,axis=0)
	out_h2_arr = np.concatenate(out_h2,axis=0)
	idx = np.argsort(out_target_arr)
	out_data_arr_sort = out_data_arr[idx,:,:,:]
	out_target_arr_sort = out_target_arr[idx]
	out_h1_arr_sort = out_h1_arr[idx,:]
	out_h2_arr_sort = out_h2_arr[idx,:]

	return out_data_arr_sort,out_target_arr_sort,out_h1_arr_sort,out_h2_arr_sort

# generate data, target and neuron activations and save them to npz file
[out_data_arr,out_target_arr,out_h1_arr,out_h2_arr] = generate_features(cuda=True)
np.savez('./results/MNIST_data_target_activations.npz', data_arr=out_data_arr, target_arr=out_target_arr, h1_arr=out_h1_arr, h2_arr=out_h2_arr)
