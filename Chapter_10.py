import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as T
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import optuna

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = get_device()
print("Using device:", device)

torch.manual_seed(42)

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

train_data, valid_data = torch.utils.data.random_split(train_and_valid_data, [55_000, 5_000])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

X_sample, y_sample = train_data[0]
#print(X_sample.shape)
#print(X_sample.dtype)



class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)

model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=188, n_hidden2=188,
                        n_classes=10).to(device)

xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0084)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # update it at each iteration
    return metric.compute()  # compute the final result at the end

evaluate_tm(model, valid_loader, accuracy)

def train(model, optimizer, criterion, metric,
          train_loader, valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

n_epochs=50

history = train(model, optimizer, xentropy, accuracy, train_loader, valid_loader,
                 n_epochs)

plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
         label="Training")
plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
         label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.title("Learning curves")
plt.axis([0.5, n_epochs + 0.5, 0.0, 1.0])
plt.legend()
plt.show()

model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit

#print(y_pred)

#def objective(trial):
#    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
#    n_hidden = trial.suggest_int("n_hidden", 20, 300)
#    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden,
#                            n_hidden2=n_hidden, n_classes=10).to(device)
#    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#    xentropy = nn.CrossEntropyLoss()
#    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
#    accuracy = accuracy.to(device)
#    history = train(model, optimizer, xentropy, accuracy, train_loader,
#                     valid_loader, n_epochs=10)
#    validation_accuracy = max(history["valid_metrics"])
#    return validation_accuracy

#torch.manual_seed(42)
#sampler = optuna.samplers.TPESampler(seed=42)
#study = optuna.create_study(direction="maximize", sampler=sampler)
#study.optimize(objective, n_trials=5)

#print(study.best_params)



