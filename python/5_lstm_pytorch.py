import torch
import torch.nn as nn
import numpy as np

# Dummy sine wave data
def generate_data(seq_len=10, n_samples=200):
    x = np.linspace(0, 100, n_samples)
    y = np.sin(x)
    X = []
    Y = []
    for i in range(len(y) - seq_len):
        X.append(y[i:i+seq_len])
        Y.append(y[i+seq_len])
    return torch.tensor(X).unsqueeze(-1).float(), torch.tensor(Y).float()

X, Y = generate_data()

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out.squeeze()

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(20):
    model.train()
    output = model(X)
    loss = loss_fn(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Final loss:", loss.item())