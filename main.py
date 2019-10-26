from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--numworkers', type=int, default=4)
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import * # data.py in the same folder
#initialize_data(args.data) # extracts the zip files, makes a validation set

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(
    [
    datasets.ImageFolder(args.data + '/train_images', transform=data_transforms),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_rotate),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_brightness),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_saturation),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_contrast),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_colorjitter_hue),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_grayscale),
    # datasets.ImageFolder(args.data + '/train_images', transform=data_transform_pad),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_centercrop),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_shear),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_hrflip),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_vrflip),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_bothflip),
    datasets.ImageFolder(args.data + '/train_images', transform=data_transform_translate),
    ]),batch_size=args.batch_size, shuffle=True, num_workers=args.numworkers, pin_memory=use_gpu)


val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=args.numworkers, pin_memory=use_gpu)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()

if use_gpu:
    model.cuda()

acc_tracker = []

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
lastlayers = list(model.fc1.parameters()) + list(model.fc2.parameters())
middlelayers = list(model.conv2.parameters())+list(model.batchnorm2.parameters())+list(model.conv3.parameters())+list(model.batchnorm3.parameters())
optimizer = optim.Adam([
        {"params" : lastlayers, "lr":1e-3},
        {"params" : middlelayers, "lr":4e-2}
    ], lr=args.lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if use_gpu:
                data = data.cuda()
                target = target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
#    scheduler.step()
    acc_tracker.append(100. * correct / len(val_loader.dataset))
    plt.plot(acc_tracker)
    plt.savefig('acc_graph.png')
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')


