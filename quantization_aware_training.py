import torch
import torchvision
from model import *
from torchvision import datasets, transforms
import time
from torch import optim


class Timer():
    def __init__(self, tag):
        self.tag = tag
        self.start = time.time()

    def __del__(self):
        self.end = time.time()
        print("{} costs {} s".format(self.tag, self.end - self.start))


def quantize_aware_training(model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, label) in enumerate(train_loader, 1):
        data , label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model.quantized_forward(data)
        loss = lossLayer(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch : {} [{}/{}] \t Loss : {:.6f}".format(epoch,
                  batch_idx * len(data), len(train_loader.dataset), loss.item()))


def full_inference(model, test_loader):
    timer = Timer("full_inference")
    correct = 0
    for idx, (data, label) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('\n Test set: Full Model Accuracy: {:.0f}%\n'.format(
        100. * correct / len(test_loader.dataset)))


def quantize_inference(model, test_loader):
    timer = Timer("quantize_inference")
    correct = 0
    for idx, (data, label) in enumerate(test_loader, 1):
        output = model.quantized_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    print('\n Test set: Quant Model Accuracy: {:.0f}%\n'.format(
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    batch_size = 64
    seed = 1
    epochs = 3
    lr = 0.01
    momentum = 0.5
    
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    model = Net()
    model.load_state_dict(torch.load('ckpt/mnist_cnn.pt', map_location='cpu'))

    model.eval()
    full_inference(model, test_loader)

    num_bits = 7
    model.quantize(num_bits=num_bits)
    
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=momentum)
    model.train()
    for epoch in range(1, epochs + 1):
        quantize_aware_training(model, device, train_loader, optimizer, epoch)
    
    model.eval()
    
    save_file = "ckpt/mnist_cnn_ptq.pt"
    torch.save(model.state_dict(), save_file)
    model.freeze()
    quantize_inference(model, test_loader)
