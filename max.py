import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet18
from utils.ConfusionMatrix import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
labels = [label for label in classes]
confusion = ConfusionMatrix(num_classes=10, labels=labels)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 128
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('model/resnet18_cifar10AdamW.pt'))
model = model.to(device)

model.eval()  # 验证模型
for data, target in valid_loader:
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    outputs = model(data)
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    confusion.update(outputs.to("cpu").numpy(), target.to("cpu").numpy())  # 更新混淆矩阵的值
confusion.plot()  # 绘制混淆矩阵
confusion.summary()  # 计算指标