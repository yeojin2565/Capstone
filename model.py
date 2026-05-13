"""
model.py

CIFAR-10 CNN 모델
Input: 3*32*32
Output: 10 classes

수정 사항:
    [BUG-3 중간] test(): loss를 배치 수로 나누지 않아 배치 수에 비례하여 부풀려지던 문제 수정.
                 loss / len(testloader) 로 평균 배치 loss 반환.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # 32×32 → pool → 16×16 → pool → 8×8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train(net, trainloader, optimizer, epochs, device):
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device):
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss   += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    # ── [BUG-3 FIX] 배치 수로 나눠 평균 loss 반환 ──────────────────
    # 수정 전: 배치 수 × 평균 loss 값이 누적됨 → 배치 수에 비례해 loss가 부풀려짐
    # 수정 후: 전체 누적 loss / 배치 수 = 평균 배치 loss (일반적인 관례)
    avg_loss = loss / len(testloader)
    return avg_loss, correct / len(testloader.dataset)