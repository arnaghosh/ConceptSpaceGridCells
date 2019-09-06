# Grid Cells in Concept Space - A deep learning perspective
This is a repository for experiments pertaining to the identification of grid cells in deep networks while learning a concept space

**Aim:** To observe if grid cell formation occurs when a network is trying to learn a concept space

**Experiment:** Train a deep network on the MNIST dataset and observe place and grid cell property in final and pre-final layers respectively

**Deep Learning Library:** Pytorch 1.1.0

## Todo
+ [X] Train a simple deep convolutional network on MNIST to identify digits
+ [ ] Obtain a tSNE representation of the MNIST dataset (testset)
+ [ ] Observe the activity pattern of final layer neurons --> ensure if it is similar to place cell-like behavior in digit space
+ [ ] Observe the activity pattern of pre-final layer neurons --> ensure if it is similar to grid cell-like behavior in digit space

## Important notes and hyperparameter considerations
* **Network architecture**: Conv-ReLU(1-->10), MaxPool(2), Conv-ReLU(10-->20), Dropout2D(0.5), MaxPool(2), FC-ReLU(320-->50), Dropout(0.5), FC(50-->10), LogSoftMax
* **Learning Rate**: 0.001
* **Optimizer**: Adam
* **Epochs**: 10
* **Batch Size**: Training = 128, Validation = 1000
* **Validation accuracy**: 98.79%
