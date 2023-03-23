import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class PytorchModel(nn.Module):
    def __init__(self):
        super(PytorchModel, self).__init__()
        self.layer1 = nn.Linear(784,512)
        self.layer2 = nn.Linear(512,10)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        y_pred = torch.softmax(x,dim=1)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建数据集
def get_datasets(train=True):
    transform = transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataloader = DataLoader(datasets.MNIST('./data', train=train, transform=transform, download=True))
    return dataloader


def train():
    epoch_num = 10
    batch_size = 200

    learning_rate = 0.01
    model = PytorchModel()
    # 获取数据集
    train_loader = get_datasets(train=True)
    test_loader = get_datasets(train=False)
    # 优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx == 100:
            #     break
            data = data.squeeze()

            optim.zero_grad()
            loss = model(data, target)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = test(model, test_loader=test_loader)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型参数
    torch.save(model.state_dict(), "model.pth")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")  # 画accuracy曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # if batch_idx == 10000:
            # break
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train()
# test()



