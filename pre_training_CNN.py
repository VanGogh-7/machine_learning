import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
import gc
import torchvision.transforms.v2 as T

REPO_ROOT = Path(__file__).resolve().parent
FLOWERS102_ROOT = REPO_ROOT / "datasets" / "flowers102"



# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            metric.update(logits, y_batch)

    return metric.compute().item()


# -----------------------------
# Training function
# -----------------------------
def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device):
    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": []
    }

    for epoch in range(n_epochs):
        model.train()
        metric.reset()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metric.update(logits, y_batch)

        train_loss = total_loss / total_samples
        train_acc = metric.compute().item()
        valid_acc = evaluate_tm(model, valid_loader, metric, device)

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_acc)
        history["valid_metrics"].append(valid_acc)

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"valid acc: {valid_acc:.4f}"
        )

    return history

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1

DefaultFlowers102 = partial(
    torchvision.datasets.Flowers102,
    root=FLOWERS102_ROOT,
    download=True
)

train_set = DefaultFlowers102(split="train", transform = transform)
valid_set = DefaultFlowers102(split="val", transform = weights.transforms())
test_set = DefaultFlowers102(split="test", transform = weights.transforms())

train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_set,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# -----------------------------
# Model
# -----------------------------
n_classes = 102

model = torchvision.models.convnext_base(weights=weights)

# Replace the final classification layer.
model.classifier[2] = nn.Linear(1024, n_classes)

model = model.to(device)


# -----------------------------
# Stage 1: Train classifier only
# -----------------------------
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

loss_fn = nn.CrossEntropyLoss()

accuracy = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=n_classes
).to(device)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

print("\nStage 1: Training classifier only")
history_stage1 = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=accuracy,
    train_loader=train_loader,
    valid_loader=valid_loader,
    n_epochs=5,
    device=device
)


# -----------------------------
# Stage 2: Fine-tune whole model
# -----------------------------
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

print("\nStage 2: Fine-tuning whole model")
history_stage2 = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=accuracy,
    train_loader=train_loader,
    valid_loader=valid_loader,
    n_epochs=5,
    device=device
)


# -----------------------------
# Final test accuracy
# -----------------------------
test_acc = evaluate_tm(model, test_loader, accuracy, device)
print(f"\nTest accuracy: {test_acc:.4f}")

