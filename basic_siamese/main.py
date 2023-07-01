import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F

# create a dataset that returns two images and a label (0 or 1) indicating whether they are from the same class or not
class SiameseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(1000, 3, 32, 32) # Generate 1000 mock RGB images of size 32x32.
        self.labels = torch.randint(0, 2, (1000,)) # Generate 1000 mock labels (0 or 1).

    def __getitem__(self, index):
        img1 = self.data[index]
        label1 = self.labels[index]

        # Find the indices where label matches label1 and where it doesn't
        positive_indices = (self.labels == label1).nonzero(as_tuple=True)[0].numpy()
        negative_indices = (self.labels != label1).nonzero(as_tuple=True)[0].numpy()

        # Choose a positive or negative example
        if label1 == 1:
            # choose a positive example
            img2_index = np.random.choice(positive_indices)
        else:
            # choose a negative example
            img2_index = np.random.choice(negative_indices)

        img2 = self.data[img2_index]

        return img1, img2, torch.tensor(int(label1 == self.labels[img2_index]), dtype=torch.float32)


    def __len__(self):
        return len(self.data)

# Create a DataLoader that returns batches of size 32
dataset = SiameseDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model, basically 2 identical subnetworks (convolutional layers followed by fully connected layers) that share weights
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5), # 3 input channels (RGB), 64 output channels, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256), # 128 * 5 * 5 is the number of features after the convolutional layers
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward_one(self, x):
        x = self.conv(x) 
        x = x.view(x.size()[0], -1) # flatten the output of the convolutional layers
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Define the contrastive loss function. This loss encourages the network to output similar features for similar images and different features for different images.
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)) 
        # the loss is used to learn similar features for similar images and different features for different images

        return loss_contrastive

model = SiameseNetwork()
optimizer = torch.optim.Adam(model.parameters()) # use Adam optimizer
criterion = ContrastiveLoss()

epochs = 10

# Train the model, for this random data and the examples is quite enough to see the loss decreasing
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        img1, img2 , label = data
        optimizer.zero_grad()
        output1, output2 = model(img1,img2)
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        
# Test the model
def inference(model, img1, img2):
    model.eval() 
    with torch.no_grad():
        output1, output2 = model(img1.unsqueeze(0), img2.unsqueeze(0)) # unsqueeze to add a batch dimension
        euclidean_distance = F.pairwise_distance(output1, output2) # calculate the euclidean distance between the two outputs
    return euclidean_distance.item()


# get items from the dataset
img1 = dataset[0][0] 
img2 = dataset[1][0] 

distance = inference(model, img1, img2)

print(f"Euclidean distance between the image pair is: {distance}")
