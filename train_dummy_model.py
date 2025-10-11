import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)


    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    

model = TinyNet()
torch.save(model.state_dict(), "tinynet.pth")