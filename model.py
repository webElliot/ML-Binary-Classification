import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__ (self):
        super(BinaryClassifier , self).__init__()
        self.layer1 = nn.Linear(93 , 192)  # Adjust the input size here
        self.layer2 = nn.Linear(192 , 128)
        self.layer3 = nn.Linear(128 , 64)
        self.layer4 = nn.Linear(64 , 32)
        self.layer5 = nn.Linear(32 , 8)
        self.layer6 = nn.Linear(8 , 1)

    def forward (self , x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = nn.functional.relu(self.layer4(x))
        x = nn.functional.relu(self.layer5(x))
        x = torch.sigmoid(self.layer6(x))
        return x