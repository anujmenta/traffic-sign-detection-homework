import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=5)
        self.batchnorm3 = nn.BatchNorm2d(250)
        self.conv4 = nn.Conv2d(250, 400, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(400)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, nclasses)

    def forward(self, x):
        bt_size = x.size(0)
        x = self.batchnorm1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        x = self.dropout(x)
        x = self.batchnorm2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        x = self.dropout(x)
        x = self.batchnorm3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        x = self.dropout(x)
        x = self.batchnorm4(F.max_pool2d(F.leaky_relu(self.conv4(x)),2))
        x = self.dropout(x)

        x = x.view(bt_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


