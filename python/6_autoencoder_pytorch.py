import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# MNIST dataset
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)

model = Autoencoder()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):
    for images, _ in loader:
        outputs = model(images)
        loss = loss_fn(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Training complete.")
# Save the model
torch.save(model.state_dict(), 'autoencoder.pth')
# Load the model
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()
# Test the model
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
import matplotlib.pyplot as plt
import numpy as np
# Visualize the results
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images)
        break
# Plot original and reconstructed images
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].numpy().squeeze(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()
# Print the first 5 predictions
print("First 5 predictions:", outputs[:5].numpy())
# Print the first 5 actual values
print("First 5 actual values:", images[:5].numpy())
# Print the first 5 features
print("First 5 features:", images[:5].view(-1, 784).numpy())