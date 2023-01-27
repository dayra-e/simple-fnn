# Autoencoder pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the data
final_df = pd.read_csv('table.csv')

# Transpose final_df
final_df = final_df.T
final_df.head(5)

# Define the model
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)
  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# hyperparameters  
learning_rate = 0.1 # learning rate for the optimizer  

# Create the model and optimizer
model = FeedForwardNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
losses = []
for i in range(final_df.shape[1]):
    # Get the inputs
    # input is the first 4 columns of the dataframe
    # label is the last 4 columns of the dataframe
    inputs = torch.from_numpy(final_df.iloc[:, :4].values).float()
    labels = torch.from_numpy(final_df.iloc[:, 4:].values).float()

    # Forward pass
    output = model(inputs)

    # Calculate the loss
    loss = torch.nn.functional.mse_loss(output, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss)

# Plot the loss
import matplotlib.pyplot as plt
# detach the loss from the graph
losses = [l.detach().numpy() for l in losses]
plt.plot(losses)
# save plt
plt.savefig('grafico-perdida.png')

# show all parameters of the model
print(model.state_dict())

# export state_dict
torch.save(model.state_dict(), 'modelo_dict.pt')
torch.save(model, 'modelo.pt')