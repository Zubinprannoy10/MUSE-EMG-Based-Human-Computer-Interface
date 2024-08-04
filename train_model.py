import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from copy import deepcopy

# Define your neural network architecture
class EMGModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EMGModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load preprocessed EMG data
# Load preprocessed EMG data
def load_data(base_dir):
    data = []
    labels = []

    for filename in os.listdir(base_dir):
        if filename.endswith(".npy"):
            # Extract the label from the filename
            label = filename.split("(")[0].strip()  # Extract the part before "(" and remove leading/trailing whitespace
            data.append(np.load(os.path.join(base_dir, filename)))
            labels.append(label)

    return data, labels

# Convert data and labels to PyTorch tensors
def convert_to_tensors(data, labels, padding_value=0):
    # Find the maximum length and number of features of the input tensors
    max_length = max(len(tensor) for tensor in data)
    max_features = max(tensor.shape[1] for tensor in data)

    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Pad the sequences with the chosen padding value to match the maximum length and number of features
    padded_data = [torch.nn.functional.pad(torch.tensor(tensor), (0, max_features - tensor.shape[1], 0, max_length - len(tensor)), mode='constant', value=padding_value) for tensor in data]

    # Convert the padded data and encoded labels to PyTorch tensors
    emg_tensor = torch.stack(padded_data, dim=0)
    labels_tensor = torch.eye(len(encoder.classes_))[encoded_labels].long()  # One-hot encode the labels and convert to LongTensor

    return emg_tensor, labels_tensor, encoder

# Create a PyTorch dataset and dataloader
def create_dataloader(emg_tensor, labels_tensor, batch_size):
    dataset = TensorDataset(emg_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Initialize the model, loss function, and optimizer
def initialize_model(input_size, output_size, hidden_size, learning_rate):
    model = EMGModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

# Train the model
def train_model(model, criterion, optimizer, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the device
    
    loss_final=0
    for epoch in range(num_epochs):
        for batch_emg, batch_labels in dataloader:
            batch_emg = batch_emg.to(device)  # Move the input tensors to the device
            batch_labels = batch_labels.to(device)  # Move the labels to the device

            # Forward pass
            outputs = model(batch_emg.float())  # Cast the input tensors to float
            loss = criterion(outputs, batch_labels)  # No need for unsqueeze(1)
            if loss.item() > loss_final:
                loss_final=deepcopy(loss.item()) 

            if loss_final>loss.item():
                
            # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Load EMG data and labels
base_dir = "./Preprocess"
data, labels = load_data(base_dir)

# Convert data and labels to PyTorch tensors
emg_tensor, labels_tensor, encoder = convert_to_tensors(data, labels)

# Create a PyTorch dataloader
batch_size = 64
dataloader = create_dataloader(emg_tensor, labels_tensor, batch_size)

# Define hyperparameters
hidden_size = 64  # Adjust according to your preference
learning_rate = 0.001
num_epochs = 1000

# Initialize the model, loss function, and optimizer
input_size = emg_tensor.shape[2]  # input_size is the last dimension of the EMG tensor
output_size = len(set(labels))  # Number of unique labels
model, criterion, optimizer = initialize_model(input_size, output_size, hidden_size, learning_rate)

# Train the model
train_model(model, criterion, optimizer, dataloader, num_epochs)

# Save the model and label encoder
torch.save(model.state_dict(), 'model.pth')
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(encoder, le_file)
