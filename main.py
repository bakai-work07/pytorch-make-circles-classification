# Import required libraries
import sklearn
from sklearn.datasets import make_circles
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# Set device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate data for binary classification
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# Convert data to tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# Split data into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network for binary classification
class CircleModelV0(nn.Module):
    def __init__(self):
        super(CircleModelV0, self).__init__()
        # 5-layer fully-connected network with ReLU activations
        self.network = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.network(x)

# Function to calculate accuracy
def accuracy_fn(y_true, y_pred):
    """Calculate classification accuracy (as percentage)."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Prepare data for training
X_train, y_train = X_train.to(device), y_train.to(device).unsqueeze(1)
X_test, y_test = X_test.to(device), y_test.to(device).unsqueeze(1)

# Train the model and save weights to file
def train_and_save_model(model, X_train, y_train, X_test, y_test, loss_fn, optimizer, epochs=3000, model_path="circle_model.pth"):
    for epoch in range(epochs):
        model.train()
        # Forward pass
        y_logits = model(X_train.squeeze())
        y_pred = torch.round(torch.sigmoid(y_logits))
        # Compute loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_train, y_pred)
        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Evaluation on test set
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test.squeeze())
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_test, test_pred)
        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Train Loss: {loss:.5f}, Train Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    # Save trained model weights to file
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

# Plot decision boundary
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Original Code Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    Modified by mrdbourke and taken from the amazing helper_functions.py"""

    # Move everything to CPU for compatibility with numpy/matplotlib
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    # Create mesh grid over feature space
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    # Get model predictions for each point in the grid
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
    # Binary classification: apply sigmoid then round
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # For multi-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))          # For binary
    # Reshape and plot decision boundary
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Reload model weights and visualize test predictions
def load_and_predict(model_class, model_path, X, y):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.inference_mode():
        y_pred = torch.round(torch.sigmoid(model(X.squeeze())))
    plot_decision_boundary(model, X, y)

if __name__ == "__main__":
    # Initiate model, loss, optimizer
    model_0 = CircleModelV0().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

    # Train and save the model
    train_and_save_model(model_0, X_train, y_train, X_test, y_test, loss_fn, optimizer, epochs=3000)

    # Load model
    preds = load_and_predict(CircleModelV0, "circle_model.pth", X_test, y_test)
