import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Net

# Set the hyperparameters and training setups, including fixing the random seeds
n_epochs = 10
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load the data
mnist_trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]))
mnist_testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3081,))]))

train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=batch_size_train,shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset,batch_size=batch_size_test,shuffle=True)

# traininig function
def train(epoch, cuda=False):
    if cuda:
        network.cuda()
    else:
        network.cpu()
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(network.state_dict(), './results/model.pth')
    torch.save(optimizer.state_dict(), './results/optimizer.pth')

# testing/evaluation function
def test(cuda=False):
    if cuda:
        network.cuda()
    else:
        network.cpu()
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Initialize model and optimizers
network = Net()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Train the model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

test(cuda=True)
for epoch in range(1, n_epochs + 1):
  	train(epoch,cuda=True)
  	test(cuda=True)

# plot the training and validation results
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()