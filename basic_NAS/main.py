import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# a function to create a model with the given architecture, dropout rate and number of layers
def create_model(num_layers, layer_sizes, dropout_rate):
    layers = []
    layers.append(nn.Flatten())  # This will flatten the input, which is a 28x28=784 image, into a 784-dimensional vector
    for i in range(num_layers):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1])) # all of our architectures are starting with 784 input features
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(layer_sizes[-1], 10))  # Final layer for classification into 10 classes, all of our architectures are ending with 10 output features because MNIST has 10 classes
    return nn.Sequential(*layers) # this is a nice way to create a sequential model, it's equivalent to nn.Sequential(layers[0], layers[1], ...)

# Define our training and validation datasets
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # this is the mean and std dev for the MNIST dataset, precalculated magic numbers
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                   transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
valid_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the layer sizes and dropout rate to explore
layer_sizes_list = [[784, 100], [784, 50, 50], [784, 200, 100, 50], [784, 300, 200, 100, 50]] # different layers sizes like [784, 100] means 784 input features and 100 output features
dropout_rates = [0.0, 0.2, 0.5] # different dropout rates

for layer_sizes in layer_sizes_list:
    for dropout_rate in dropout_rates:
        # Create a model with the current architecture
        model = create_model(len(layer_sizes)-1, layer_sizes, dropout_rate).to(device)
        
        # Use cross entropy loss for this classification task
        loss_fn = nn.CrossEntropyLoss()

        # Use stochastic gradient descent as the optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train the model for a few epochs (we'll just use 1 here for simplicity)
        for epoch in range(1):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Reset the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

        # Evaluate the model on the validation data
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Accuracy of the model with architecture {layer_sizes} and dropout rate {dropout_rate}: {100 * correct / total}%")

# for my runs the results were:
# Accuracy of the model with architecture [784, 100] and dropout rate 0.0: 90.53%
# Accuracy of the model with architecture [784, 100] and dropout rate 0.2: 89.22%
# Accuracy of the model with architecture [784, 100] and dropout rate 0.5: 86.72%
# Accuracy of the model with architecture [784, 50, 50] and dropout rate 0.0: 88.79%
# Accuracy of the model with architecture [784, 50, 50] and dropout rate 0.2: 82.85%
# Accuracy of the model with architecture [784, 50, 50] and dropout rate 0.5: 68.67%
# Accuracy of the model with architecture [784, 200, 100, 50] and dropout rate 0.0: 87.75%
# Accuracy of the model with architecture [784, 200, 100, 50] and dropout rate 0.2: 77.35%
# Accuracy of the model with architecture [784, 200, 100, 50] and dropout rate 0.5: 56.78%
# Accuracy of the model with architecture [784, 300, 200, 100, 50] and dropout rate 0.0: 68.18%
# Accuracy of the model with architecture [784, 300, 200, 100, 50] and dropout rate 0.2: 58.64%
# Accuracy of the model with architecture [784, 300, 200, 100, 50] and dropout rate 0.5: 25.13%