import torch
import torchvision
import torchvision.transforms as transforms


'''
torchvision中提供了主流的公开数据集，如MNIST、CIFAR-10、CIFAR-100等。
MNIST（手写黑白图像数据集）比较小并且简单，并不能很好地体现现在模型的性能。
使用CIFAR-10包含了共十类彩色图像数据集），可以较好地测试模型的能力。
CIFAR-100更加困难一些。
torchvision.transforms可以对图像进行数据增强，并使用transforms.ToTensr()将数据转换成训练所需要的tensor
'''

transform = transforms.Compose([transforms.transforms.RandomRotation(0.5),
                                transforms.RandomGrayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data/image',
                                             train=True,
                                             download=True,
                                             transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = 50,
                                           shuffle = True,
                                           num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data/image',
                                            train=False,
                                            download=False,
                                            transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=25,
                                          shuffle=False,
                                          num_workers=2)


# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(train_dataset.data[0]) # 第一个数据
# plt.show()