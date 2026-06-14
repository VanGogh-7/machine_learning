import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        def forward(self, x):
            x =x.view(-1, 2048)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            