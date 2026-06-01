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
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import urllib.request
from datasets import load_dataset


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

imdb_dataset = load_dataset("imdb")
split = imdb_dataset["train"].train_test_split(train_size=0.8, seed=42)
imdb_train_set, imdb_valid_set = split["train"], split["test"]
imdb_test_set = imdb_dataset["test"]


