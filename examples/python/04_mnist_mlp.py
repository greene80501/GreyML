"""GreyML example: 04 mnist mlp.
Shows a minimal usage pattern you can copy into your own experiments.
"""

import sys
sys.path.insert(0, "python")

from greyml import Tensor, nn, optim
from greyml.data import DataLoader
import numpy as np

# Simple MLP for MNIST
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dummy data
X_train = Tensor(np.random.randn(100, 784))
y_train = Tensor(np.random.randint(0, 10, 100))

# Training
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.functional.mse_loss

for epoch in range(2):
    total_loss = 0
    for i in range(0, 100, 32):
        X_batch = X_train[i:i+32]
        y_batch = y_train[i:i+32]
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")