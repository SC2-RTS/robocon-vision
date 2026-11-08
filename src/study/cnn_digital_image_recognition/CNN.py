import torch, time, cv2
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def create_dataset():
    train_dataset = MNIST(root='./data', train=True,  transform=ToTensor(), download=True)
    test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

    return train_dataset, test_dataset

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        self.linear1 = nn.Linear(784, 240)
        self.linear2 = nn.Linear(240, 84)
        self.output = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        x = self.output(x)

        return x

def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ImageModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs, total_time = 5, time.time()
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0, 0, 0, time.time()
        for x, y in dataloader:
            model.train()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        print(f'epoch: {epoch + 1}, loss: {total_loss / total_samples:.4f}, acc: {total_correct / total_samples:.2f}, time: {time.time() - start:.2f}s')

    print(f'total time: {time.time() - total_time:.2f}s')
    torch.save(model.state_dict(), './model/digital_image_model.pth')

def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ImageModel().to(device)
    model.load_state_dict(torch.load('./model/digital_image_model.pth'))

    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            model.eval()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_samples += len(y)

    print(f'acc: {total_correct / total_samples:.2f}')

if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()

    # model = ImageModel().to(device)
    # summary(model, (1, 28, 28))

    # train(train_dataset)

    evaluate(test_dataset)
    