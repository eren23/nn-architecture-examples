import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Number of users and items
num_users = 5000
num_items = 1000

# Number of features for users and items
num_user_features = 50
num_item_features = 100

# Generate dummy data
user_features = np.random.rand(num_users, num_user_features)
item_features = np.random.rand(num_items, num_item_features)

# We can also generate random interactions between users and items
interactions = np.random.randint(0, 2, size=(num_users, num_items))

class TwoTowerNetwork(nn.Module):
    def __init__(self, num_user_features, num_item_features, embedding_dim):
        super(TwoTowerNetwork, self).__init__()
        
        # User tower, is smaller than item tower, because we have fewer user features
        self.user_tower = nn.Sequential(
            nn.Linear(num_user_features, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(num_item_features, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    # Forward pass
    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        return user_embedding, item_embedding

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
embedding_dim = 32 # Dimensionality of the embeddings, for this task 32 is plenty
model = TwoTowerNetwork(num_user_features, num_item_features, embedding_dim).to(device)

# Define the loss function and optimizer
loss_function = nn.CosineEmbeddingLoss()

# We use the Adam optimizer with default parameters, but feel free to experiment with different optimizers
optimizer = Adam(model.parameters())

class InteractionsDataset(Dataset):
    """PyTorch Dataset class for user-item interactions."""

    # The __init__ function is run once when instantiating the InteractionsDataset object.
    def __init__(self, user_features, item_features, interactions):
        self.user_features = user_features
        self.item_features = item_features
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    # 
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        user_feature = self.user_features[interaction[0]]
        item_feature = self.item_features[interaction[1]]
        label = interaction[2]
        return user_feature, item_feature, label

# Create a list of (user, item, interaction) tuples
interactions_list = [(user_id, item_id, interactions[user_id, item_id]) for user_id in range(num_users) for item_id in range(num_items)]

# Split data into training and validation sets
train_interactions, val_interactions = train_test_split(interactions_list, test_size=0.2, random_state=42)

# Create PyTorch Datasets and DataLoaders
batch_size = 32
train_dataset = InteractionsDataset(user_features, item_features, train_interactions)
val_dataset = InteractionsDataset(user_features, item_features, val_interactions)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # Training
    model.train()
    for user_features_batch, item_features_batch, labels_batch in train_loader:
        # Convert data to PyTorch tensors
        user_features_batch = torch.tensor(user_features_batch, dtype=torch.float32).to(device)
        item_features_batch = torch.tensor(item_features_batch, dtype=torch.float32).to(device)
        labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)

        # Compute embeddings and loss
        user_embeddings, item_embeddings = model(user_features_batch, item_features_batch)
        
        # CosineEmbeddingLoss, quite simply, takes two embeddings and a label as input, and computes the cosine similarity between the embeddings.
        loss = loss_function(user_embeddings, item_embeddings, labels_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        for user_features_batch, item_features_batch, labels_batch in val_loader:
            # Convert data to PyTorch tensors
            user_features_batch = torch.tensor(user_features_batch, dtype=torch.float32).to(device)
            item_features_batch = torch.tensor(item_features_batch, dtype=torch.float32).to(device)
            labels_batch = torch.tensor(labels_batch, dtype=torch.float32).to(device)

            # Compute embeddings and loss
            user_embeddings, item_embeddings = model(user_features_batch, item_features_batch)
            loss = loss_function(user_embeddings, item_embeddings, labels_batch)

            epoch_val_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss = {epoch_train_loss/len(train_loader)}, Val Loss = {epoch_val_loss/len(val_loader)}')
    

# Inference code below, to use the model for making recommendations
# Compute the embeddings for all users and items
with torch.no_grad():
    user_embeddings = model.user_tower(torch.tensor(user_features, dtype=torch.float32).to(device))
    item_embeddings = model.item_tower(torch.tensor(item_features, dtype=torch.float32).to(device))

# Convert the embeddings to numpy arrays
user_embeddings = user_embeddings.cpu().numpy()
item_embeddings = item_embeddings.cpu().numpy()

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(user_embeddings, item_embeddings)

# Make recommendations for each user
num_recommendations = 10
for user_id in range(num_users):
    # Get the similarity scores for this user
    similarity_scores = similarity_matrix[user_id]
    
    # Get the indices of the items with the highest similarity scores
    recommended_item_ids = np.argsort(similarity_scores)[-num_recommendations:]
    
    print(f'Recommended items for user {user_id}: {recommended_item_ids}')