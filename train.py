import torch
import torchvision
from torch import optim
from torchvision import datasets, transforms
from model import *

import os
import os.path as osp

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print("Train Epoch : {} [{}/{}] \t Loss : {:.6f}".format(epoch,
                  batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, device, test_loader):
    model.eval()
    loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss += lossLayer(output, label).item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    loss /= len(test_loader.dataset)
    print('\n Test Set : Average loss : {:.4f} , Accuracy : {:.0f} %\n'.format(
        loss, 100. * correct/len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epoches = 15
    lr = 0.01
    momentum = 0.5

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
                                               transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epoches + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if not osp.exists("ckpt") :
        os.makedirs("ckpt")
    torch.save(model.state_dict(), 'ckpt/mnist_cnn.pt')