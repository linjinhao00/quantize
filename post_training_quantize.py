import torch
import torchvision
from model import *
from torchvision import datasets, transforms
import time


class Timer():
    def __init__(self, tag):
        self.tag = tag
        self.start = time.time()

    def __del__(self):
        self.end = time.time()
        print("{} costs {} s".format(self.tag, self.end - self.start))


def direct_quantize(model, train_loader):
    for i, (data, label) in enumerate(test_loader, 1):
        output = model.quantized_forward(data)
        if i % 500 == 0:
            break
    print("driect quantization finish")


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
    model.eval()

    direct_quantize(model, train_loader)
    save_file = "ckpt/mnist_cnn_ptq.pt"
    torch.save(model.state_dict(), save_file)
    model.freeze()
    quantize_inference(model, test_loader)
