# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define the model, basically 2 identical subnetworks (convolutional layers followed by fully connected layers) that share weights
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5),  # Convolution layer: 1 input channel, 64 output channels, kernel size 5
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(64, 128, 5),  # Another convolution layer: 64 input channels, 128 output channels, kernel size 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # Fully connected layer: 128*4*4 input features, 256 output features
            nn.ReLU(),  
            nn.Linear(256, 256),  
            nn.ReLU(), 
            nn.Linear(256, 2)  # Final fully connected layer: 256 input features, 2 output features
        )

    def forward_one(self, x):
        x = self.conv(x)  
        x = x.view(x.size()[0], -1) 
        x = self.fc(x)  
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# A custom dataset for creating Siamese pairs from MNIST data
class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset  # Store the MNIST dataset
        self.train_labels = self.mnist_dataset.targets  
        self.train_data = self.mnist_dataset.data 
        self.labels_set = set(self.train_labels.numpy())  
       
        self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]  for label in self.labels_set} # Create a dictionary mapping each label to the indices where it appears

    def __getitem__(self, index):
        target = np.random.randint(0, 2)  # Randomly choose a target label (0 or 1)
        img1, label1 = self.train_data[index], self.train_labels[index].item()  # Get the image and label at the chosen index
        if target == 1:  # If the target is 1, we want to find another image with the same label
            siamese_index = index
            while siamese_index == index:  # Choose another index for the same label
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:  # If the target is 0, we want to find an image with a different label
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))  # Choose a different label
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])  # Choose an index for the different label
        img2 = self.train_data[siamese_index]  # Get the second image
        img1 = img1/255.0  
        img2 = img2/255.0 
        img1 = img1.unsqueeze(0).float() 
        img2 = img2.unsqueeze(0).float()
        return (img1, img2), target  # Return the pair of images and the target

    def __len__(self):
        return len(self.mnist_dataset)  # Return the size of the dataset

# Contrastive loss function for the Siamese network
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin  # Define the margin for the contrastive loss

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)  # Compute the Euclidean distance between the two outputs
        # Compute the contrastive loss
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive 
    
# Load the MNIST dataset
train_data = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_data = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Create a DataLoader that returns pairs of images and labels
siamese_train_dataset = SiameseMNIST(train_data)
siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=32, shuffle=True)

model = SiameseNetwork()
optimizer = torch.optim.Adam(model.parameters())
criterion = ContrastiveLoss()

# Train the model
epochs = 10
for epoch in range(epochs):
    for i, data in enumerate(siamese_train_loader, 0):
        (img1, img2) , label = data  
        optimizer.zero_grad()
        output1, output2 = model(img1,img2) 
        label = label.float() 
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()  
        optimizer.step()

# Test the model
def inference(model, img1, img2):
    model.eval()  
    with torch.no_grad():  
        output1, output2 = model(img1.unsqueeze(0), img2.unsqueeze(0))  
        euclidean_distance = F.pairwise_distance(output1, output2) 
    return euclidean_distance.item()  

# Get a pair of images to test the model
img1 = train_data[0][0] 
img2 = train_data[1][0] 

distance = inference(model, img1, img2)

print(f"Euclidean distance between the image pair is: {distance}") 