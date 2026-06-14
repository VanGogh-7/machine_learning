import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import gc

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

current_dir = Path(__file__).resolve().parent
image_path = current_dir.parent / "data" / "coffee.jpg"
img = Image.open(image_path)

print("PIL image information")
print("Format:", img.format)
print("Mode:", img.mode)
print("Size:", img.size)  # (width, height)

img_array = np.array(img)

print("\nNumPy array information")
print("Shape:", img_array.shape)  # (height, width, channels)
print("Dtype:", img_array.dtype)
print("Min pixel value:", img_array.min())
print("Max pixel value:", img_array.max())

plt.imshow(img)
plt.axis("on")
#plt.show()

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                ])
img_tensor = transform(img)
print(type(img_tensor), img_tensor.shape)

batch = img_tensor.unsqueeze(0)
print(type(batch), batch.shape)

weights = models.AlexNet_Weights.DEFAULT
model = models.alexnet(weights=weights)

model.eval()
model.to(device)

batch = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    y = model(batch)

print("type and shape of y:", type(y), y.shape)
probs = torch.softmax(y, dim=1)
print(probs.shape)

predicted_class = probs.argmax(dim=1)
print(predicted_class)

